# fontaine/ — the autonomous research agent

Everything that defines and runs **Fontaine**, the autonomous Bijou
research agent, lives in this directory. `charter.md` is the
governing document — read it first. This README is the owner-side
runbook (ignition + day-to-day) and the map of the directory.

| path | what |
|---|---|
| `charter.md` | the charter: mission, metric, discipline, boundaries, agenda |
| `prompts/bootstrap.md` | the first-ever session: executes charter §10 |
| `prompts/tick.md` | the 30-minute babysit tick |
| `prompts/work.md` | a bounded work session (analysis, launches, writing) |
| `harness/fontaine-session.sh` | runs ONE headless session (`tick`\|`work`\|`bootstrap`) |
| `harness/discord.py` | the Discord "tool": REST read/post + cursor (stdlib, no deps) |
| `harness/systemd/` | user units: the 30-min tick timer |
| `harness/state/` | session lock + Discord cursor (gitignored) |
| `harness/logs/` | session transcripts (gitignored) |
| `blog/` | the mdbook lab notebook (created by the agent at bootstrap; built site gitignored) |

## Ignition (owner, once)

1. **Box** — initialized by you before Fontaine ever runs: a 1×H100
   Lambda instance (≥2 TB disk), `./init-vm-gpu.sh` (repo lands at
   `~/flow-matching`), HF + wandb auth, and the datasets staged on
   disk under `~/datasets/mcobzarenco/` — `community_curated_v0` and
   the two `so101_pick_place_{v2,clean}` rig repos. Fontaine verifies
   all of this at bootstrap; it performs none of it and never
   re-downloads staged data.
2. **Branch** — create the agent's branch from `main` and track it on
   the box:

   ```sh
   cd ~/flow-matching
   git fetch && git checkout -b fontaine origin/main
   git push -u origin fontaine
   ```

3. **Git push credential** — Fontaine's sessions fire from a systemd
   timer with no SSH session and no forwarded agent, so the box
   needs its own key. A **repo-scoped deploy key** fits the safety
   model (one repo, nothing else on the account):

   ```sh
   ssh-keygen -t ed25519 -f ~/.ssh/id_fontaine -N "" -C "fontaine deploy key (flow-matching)"
   printf 'Host github.com\n    IdentityFile ~/.ssh/id_fontaine\n    IdentitiesOnly yes\n' > ~/.ssh/config
   chmod 600 ~/.ssh/config
   cd ~/flow-matching
   git config user.name "Fontaine"
   git config user.email "fontaine-agent@users.noreply.github.com"
   ```

   Add `~/.ssh/id_fontaine.pub` on GitHub: repo → Settings → Deploy
   keys → Add deploy key → check **Allow write access**. Verify with
   `ssh -T git@github.com` (a deploy key greets with the repo name,
   not a username). Two notes: deploy keys cannot be branch-limited,
   so "never push to `main`" stays contractual (charter §7); and
   with `IdentitiesOnly`, interactive pushes from the box use this
   key too.
4. **Discord** — the auth model is a *bot token over plain REST*
   (`harness/discord.py` is the whole integration):
   - discord.com/developers → **New Application** → **Bot** →
     **Reset Token** → copy the bot token (treat it as a password).
   - Same Bot page → Privileged Gateway Intents → enable the
     **Message Content intent** — without it the bot receives EMPTY
     message bodies over both gateway and REST.
   - OAuth2 → URL Generator → scope `bot`; permissions **View
     Channel**, **Send Messages**, **Read Message History** → open
     the generated URL and invite the bot to your private server's
     `#fontaine` channel.
   - In your client: Settings → Advanced → **Developer Mode** on;
     right-click the channel → **Copy Channel ID**; right-click your
     avatar → **Copy User ID** (used for @mention escalations).
5. **Claude Code** — install and authenticate on the box:

   ```sh
   curl -fsSL https://claude.ai/install.sh | bash   # or: npm install -g @anthropic-ai/claude-code
   claude --version
   cd ~/flow-matching && claude                     # first run: complete the login
   ```

   The interactive login prints a URL — open it in your laptop
   browser, authorize, paste the code back into the SSH session;
   credentials persist on the box for the headless sessions.
   Alternatives: `claude setup-token` (mints a long-lived token for
   headless boxes), or an `ANTHROPIC_API_KEY` in the env file
   (API billing instead of the subscription).
6. **Env file** `~/.config/fontaine/env` (mode 600, never committed):

   ```sh
   DISCORD_BOT_TOKEN=...
   DISCORD_CHANNEL_ID=...
   DISCORD_OWNER_ID=...         # optional, for @mentions
   WANDB_API_KEY=...            # shared account key; project `fontaine`
   FONTAINE_MODEL=fable         # pin the agent's model (→ claude --model);
                                # verify the resolved id after auth (below)
   # MAX_THINKING_TOKENS=31999  # harness default; raise only if the
                                # CLI/model accepts more
   ```

   Pinning model + thinking: `FONTAINE_MODEL` is passed to every
   session as `--model`; the harness exports
   `MAX_THINKING_TOKENS=31999` (Claude Code's max "ultrathink"
   budget) unless the env file overrides it. After authenticating,
   verify the pin resolves to Fable 5:

   ```sh
   claude -p "say OK" --model fable --output-format json | grep -o "claude[a-z0-9.-]*" | sort -u
   ```

   (or run `/model` inside interactive `claude` to list what the
   account offers and the exact id — use that id in `FONTAINE_MODEL`
   if the `fable` alias doesn't resolve).

7. **Start it**:

   ```sh
   ~/flow-matching/fontaine/harness/fontaine-session.sh bootstrap
   ```

   The bootstrap session verifies every credential and the staged
   datasets with measured checks, builds the blog + Space, enables
   the tick timer, scores the baseline, and introduces itself in
   Discord. From there the agent takes over (charter §10) — the
   timer keeps it alive indefinitely.

   (Timer enablement, for reference or manual repair:
   `ln -sf ~/flow-matching/fontaine/harness/systemd/fontaine-tick.* ~/.config/systemd/user/`,
   `systemctl --user daemon-reload`,
   `systemctl --user enable --now fontaine-tick.timer`, and
   `sudo loginctl enable-linger ubuntu` so it fires without an SSH
   session.)

## How the Discord "tool" works

There is no special integration, no gateway connection, no MCP
server: `harness/discord.py` (Python stdlib only) wraps two REST
calls with the bot token from the env, and headless sessions run it
via bash like any other command:

```sh
uv run python fontaine/harness/discord.py read          # new messages since the cursor
uv run python fontaine/harness/discord.py post "text"   # post to #fontaine
```

`read` keeps a cursor (the last-seen message id) in `harness/state/`,
so each tick sees exactly what arrived since the previous tick; the
first read initializes the cursor at the channel head without
replaying backlog (stale steering must never surface as new).
Polling at tick cadence deliberately replaces a persistent gateway
connection — nothing to keep alive, nothing to crash.

## Day-to-day (owner)

- **Steer by messaging `#fontaine`** — every tick polls the channel;
  steering overrides the agenda (charter §7).
- **Read the blog** for substance:
  `https://huggingface.co/spaces/mcobzarenco/fontaine-blog`
  (`now.md` = what the GPU is doing this hour and why).
- The agent never pushes to `main`; adopt findings by cherry-pick
  from the `fontaine` branch when a write-up earns it.

## Safety model

Credential scope IS the blast radius: the HF token writes only
`fontaine-*`, wandb is scoped to its project by convention, the
Discord bot sees one channel, the git deploy key writes one
repository (branch discipline is contractual — deploy keys cannot be
branch-limited). The box is
single-purpose, so headless sessions run with tool permissions
skipped *inside* that scope — the charter's §7 boundaries are
contractual, and the credentials make the important ones physical.
