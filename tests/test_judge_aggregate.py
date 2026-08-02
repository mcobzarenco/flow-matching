"""Pure-CPU tests for the calibration aggregator's decision logic."""

from __future__ import annotations

import json
from pathlib import Path

from bijou.judge.aggregate import (
    EpisodeVerdict,
    compare_models,
    load_verdicts,
    majority_camera_vote,
    review_worksheet,
    rule_impact,
)
from bijou.judge.schema import (
    PROMPT_HASH,
    CameraKind,
    CameraVisibility,
    EpisodeJudgment,
    FrameAnnotation,
    InstructionQuality,
    Scores,
    Subgoal,
    TaskCompletion,
    Verdict,
)


def judgment(
    *,
    verdict: Verdict = Verdict.KEEP,
    overall: int = 7,
    completion: TaskCompletion = TaskCompletion.YES,
    camera_kind: CameraKind = CameraKind.FRONT,
    holding: tuple[bool, ...] = (False, True),
    progress: tuple[float, ...] = (0.0, 1.0),
) -> EpisodeJudgment:
    annotations = tuple(
        FrameAnnotation(
            frame=1 + 10 * i,
            progress=progress[i],
            holding=holding[i],
            visible={"cam": CameraVisibility(task_object=True, gripper=True)},
            events=(),
        )
        for i in range(len(holding))
    )
    return EpisodeJudgment(
        overall_score=overall,
        verdict=verdict,
        task_completion_visible=completion,
        scores=Scores(visual_quality=7, smoothness=7, efficiency=7, camera_framing=7),
        instruction_quality=InstructionQuality.GOOD,
        observed_task="a task",
        suggested_instructions=("do the task",),
        subgoals=(Subgoal(until_frame=11, subgoal="do it"),),
        frame_annotations=annotations,
        camera_kinds={"cam": camera_kind},
        issues=(),
        summary="",
    )


def test_majority_vote_tie_resolves_to_unknown() -> None:
    vote = majority_camera_vote([CameraKind.FRONT, CameraKind.TOP])
    assert vote.kind is CameraKind.UNKNOWN
    assert vote.tie and not vote.unanimous
    assert vote.votes == {"front": 1, "top": 1}


def test_majority_vote_majority_wins() -> None:
    vote = majority_camera_vote([CameraKind.TOP, CameraKind.TOP, CameraKind.FRONT])
    assert vote.kind is CameraKind.TOP
    assert not vote.tie and not vote.unanimous


def test_majority_vote_unanimous() -> None:
    vote = majority_camera_vote([CameraKind.WRIST, CameraKind.WRIST])
    assert vote.kind is CameraKind.WRIST
    assert vote.unanimous and not vote.tie


def test_rule_impact_dataset_thresholds() -> None:
    verdicts = [
        EpisodeVerdict("u/a", 0, "m", judgment(verdict=Verdict.DISCARD)),
        EpisodeVerdict("u/a", 1, "m", judgment(verdict=Verdict.KEEP)),
        EpisodeVerdict("u/b", 0, "m", judgment(verdict=Verdict.DISCARD)),
        EpisodeVerdict("u/b", 1, "m", judgment(verdict=Verdict.DISCARD)),
        EpisodeVerdict("u/c", 0, "m", judgment(verdict=Verdict.KEEP)),
    ]
    impact = rule_impact(verdicts, lambda j: j.verdict is Verdict.DISCARD)
    assert impact.episodes_flagged == 3
    assert impact.datasets_half_flagged == 2  # a (1/2) and b (2/2)
    assert impact.datasets_all_flagged == 1  # b only


def test_compare_models_pairs_by_episode_and_frame() -> None:
    verdicts = [
        EpisodeVerdict("u/a", 0, "opus", judgment(overall=8)),
        EpisodeVerdict(
            "u/a",
            0,
            "haiku",
            judgment(
                verdict=Verdict.REVIEW,
                overall=6,
                camera_kind=CameraKind.TOP,
                holding=(False, False),
                progress=(0.0, 0.5),
            ),
        ),
        # Unpaired episodes must not contribute.
        EpisodeVerdict("u/a", 1, "opus", judgment()),
        EpisodeVerdict("u/b", 0, "haiku", judgment()),
    ]
    comparison = compare_models("opus", "haiku", verdicts)
    assert comparison.episodes == 1
    assert comparison.verdict_agreement == 0.0
    assert comparison.verdict_confusion["keep"]["review"] == 1
    assert comparison.overall_mae == 2.0
    assert comparison.holding_frames == 2
    assert comparison.holding_agreement == 0.5  # frame 1 agrees, frame 11 not
    assert abs(comparison.progress_mae - 0.25) < 1e-9
    assert comparison.cameras_compared == 1
    assert comparison.camera_kind_agreement == 0.0
    # Single paired episode: correlation undefined, not fabricated.
    assert comparison.overall_pearson is None


def test_load_verdicts_latest_wins_across_evidence_sizes(tmp_path: Path) -> None:
    """The store keys on evidence parameters, so one (episode, model) can
    hold several records; aggregation must collapse to the latest."""
    dataset_dir = tmp_path / "u" / "ds"
    (dataset_dir / "meta").mkdir(parents=True)
    record = {
        "episode_index": 0,
        "model": "m",
        "prompt_hash": PROMPT_HASH,
        "num_timesteps": 10,
        "max_image_dim": 512,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    older = {
        **record,
        "judged_at": "2026-08-01 00:00:00",
        "judgment": judgment(overall=3).to_dict(),
    }
    newer = {
        **record,
        "num_timesteps": 15,
        "judged_at": "2026-08-02 00:00:00",
        "judgment": judgment(overall=9).to_dict(),
    }
    (dataset_dir / "meta" / "judgments.json").write_text(
        json.dumps({"judgments": [newer, older]}),
    )
    verdicts, datasets, skipped = load_verdicts([dataset_dir], PROMPT_HASH)
    assert (datasets, skipped) == (1, 0)
    assert len(verdicts) == 1
    assert verdicts[0].judgment.overall_score == 9  # latest judged_at won


def test_review_worksheet_stratifies_and_is_seeded() -> None:
    verdicts = [
        EpisodeVerdict(
            f"u/d{i}",
            i,
            "m",
            judgment(verdict=verdict_kind),
        )
        for verdict_kind, count in (
            (Verdict.KEEP, 20),
            (Verdict.REVIEW, 3),
            (Verdict.DISCARD, 3),
        )
        for i in range(count)
    ]
    sheet = review_worksheet(verdicts, sample_size=9, seed=0)
    assert len(sheet) == 9
    by_verdict = dict.fromkeys(("keep", "review", "discard"), 0)
    for row in sheet:
        by_verdict[row["judge_verdict"]] += 1
        assert row["human_verdict"] == ""
    assert by_verdict["review"] == 3
    assert by_verdict["discard"] == 3
    assert by_verdict["keep"] == 3
    assert sheet == review_worksheet(verdicts, sample_size=9, seed=0)
    assert sheet != review_worksheet(verdicts, sample_size=9, seed=1)
