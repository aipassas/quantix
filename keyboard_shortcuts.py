"""Keyboard shortcuts and a command palette, on a platform that officially
cannot have them.

WHY THIS IS POSSIBLE AT ALL, given onboarding.py says the opposite.
onboarding.py is right about what it describes: st.markdown(...,
unsafe_allow_html=True) inserts HTML via innerHTML, and a <script> tag
inserted that way never runs. That is a real wall and this app has hit
it twice.

st.components.v1.html is a different mechanism — a real iframe with a
real document, where scripts DO execute. The question that decides
everything is whether that iframe is same-origin, because a cross-origin
one cannot see the page it sits in. It is: probed on the running app,
window.parent.document came back ACCESSIBLE, a keydown listener attached
to it, and a synthetic click on a parent button drove a genuine Streamlit
rerun. So the bridge is: iframe listens for the key -> iframe clicks a
hidden Streamlit button in the parent -> Streamlit reruns with new state.
No bidirectional component, no third-party dependency.

WHICH KEYS THE BROWSER ACTUALLY LETS US HAVE, measured rather than
assumed. "?", Cmd+K, Cmd+1..9 and Cmd+/ all arrive at the page and can be
preventDefault()ed. Cmd+N does NOT — Chrome reserves it for New Window
and the page never sees the event, so the task's "Cmd+N for new alert" is
not deliverable as written. Cmd+Shift+A was probed as an alternative,
arrives cleanly, and is a better mnemonic for Alert anyway.

TABS ARE CLICKED DIRECTLY, not routed through Python. A Streamlit tab is
a real button in the parent DOM, so Cmd+1 can activate it with no server
round trip — instant, and it avoids a rerun that would reset scroll. The
strip is found by its FIRST TAB'S LABEL rather than by position, because
this page has several nested tab groups and "the first .stTabs" would be
a coin flip.

BARE KEYS YIELD TO TYPING. "?" is a character someone types into the
ticker box; a shortcut that steals it would be a bug. Bare-key bindings
are skipped whenever the event target is an input, textarea, select or
contenteditable. Modifier combos still fire there, which is what a user
pressing Cmd+K mid-sentence expects.
"""
import json
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

# The main analysis tabs, in the order finance.py creates them. A test
# asserts this matches the real st.tabs([...]) call — two lists of the
# same eight labels would drift the first time one is renamed.
MAIN_TABS: Tuple[str, ...] = (
    "Overview", "Chart Workspace", "Fundamentals & Valuation",
    "Risk & Technicals", "Monte Carlo & Seasonality", "Smart Money & Peers",
    "Portfolio", "CIO Tear Sheet",
)

# Widget keys for the hidden buttons the listener clicks. Values are the
# st.button keys; the JS finds them by their st-key-<key> container.
TRIGGERS: Dict[str, str] = {
    "palette": "kbd_trigger_palette",
    "shortcuts": "kbd_trigger_shortcuts",
    "help": "kbd_trigger_help",
    "new_alert": "kbd_trigger_new_alert",
    "close": "kbd_trigger_close",
}

# One hidden button per asset-class pill, clicked by the Alt+N binding.
# Shares the kbd_trigger_ prefix so finance.py's existing hiding rule
# covers these too.
ASSET_TRIGGER_PREFIX = "kbd_trigger_asset_"


@dataclass(frozen=True)
class Shortcut:
    keys: str            # as shown to the reader, e.g. "⌘K"
    description: str
    category: str = "General"
    note: str = ""       # shown in the overlay when there is a caveat


SHORTCUTS: Tuple[Shortcut, ...] = (
    Shortcut("?", "Show this shortcuts panel", "General",
             "Ignored while you are typing in a field."),
    Shortcut("⌘K / Ctrl+K", "Open the command palette", "General"),
    Shortcut("⌘/ / Ctrl+/", "Open Help & Support", "General"),
    Shortcut("Esc", "Close the palette or this panel", "General"),
    Shortcut("⌘1 – ⌘8", "Jump to an analysis tab", "Navigation",
             "In tab order: " + ", ".join(
                 f"{i + 1} {name}" for i, name in enumerate(MAIN_TABS))),
    Shortcut("⌘⇧A", "Create an alert for the current ticker", "Actions",
             "The task asked for ⌘N; Chrome reserves that for New Window "
             "and the page never receives it."),
    Shortcut("Alt+1 – Alt+6", "Switch asset class", "Navigation",
             "Probed on the running page: Alt+digit arrives and is "
             "preventable. Matched on the key's POSITION (event.code), "
             "because macOS composes Option+1 into ¡ — matching on "
             "event.key would fire on Windows and never on a Mac."),
)


@dataclass(frozen=True)
class Command:
    id: str
    label: str
    kind: str                       # "tab" | "action"
    keywords: Tuple[str, ...] = ()
    hint: str = ""                  # the shortcut, if it has one
    payload: Optional[int] = None   # tab index, for kind="tab"


def _tab_commands(tab_labels: Optional[Sequence[str]] = None) -> Tuple[Command, ...]:
    """`tab_labels` is the strip currently on screen.

    Tab LABELS follow the asset class (a fund's third tab reads "Holdings
    & Fund Profile", not "Fundamentals & Valuation"), so a palette built
    from the module constant would offer to jump to a tab whose name is
    not on the page. MAIN_TABS remains the fallback and the equity set.
    """
    names = tuple(tab_labels) if tab_labels else MAIN_TABS
    return tuple(
        Command(id=f"tab:{i}", label=f"Go to {name}", kind="tab",
                keywords=tuple(name.lower().replace("&", " ").split()),
                hint=f"⌘{i + 1}" if i < 9 else "", payload=i)
        for i, name in enumerate(names)
    )


ACTION_COMMANDS: Tuple[Command, ...] = (
    Command("action:new_alert", "Create an alert for this ticker", "action",
            ("alert", "notify", "watch", "rule", "new"), "⌘⇧A"),
    Command("action:help", "Open Help & Support", "action",
            ("help", "support", "docs", "question"), "⌘/"),
    Command("action:shortcuts", "Show keyboard shortcuts", "action",
            ("keyboard", "shortcuts", "keys", "bindings"), "?"),
)


def commands(tab_labels: Optional[Sequence[str]] = None) -> Tuple[Command, ...]:
    return _tab_commands(tab_labels) + ACTION_COMMANDS


def search(query: str, pool: Optional[Sequence[Command]] = None) -> Tuple[Command, ...]:
    """Commands matching `query`, best first.

    Plain case-insensitive substring matching over the label and the
    keywords, ranked by where the match lands. Deliberately not fuzzy:
    a palette that reorders results by an opaque score is impossible to
    predict, and this list is short enough that substring matching finds
    everything a person would type.
    """
    pool = tuple(pool) if pool is not None else commands()
    query = (query or "").strip().lower()
    if not query:
        return pool

    scored: List[Tuple[int, int, Command]] = []
    for index, command in enumerate(pool):
        label = command.label.lower()
        position = label.find(query)
        if position >= 0:
            # A label match beats a keyword match; earlier beats later.
            scored.append((0 if position == 0 else 1, position, command))
            continue
        if any(query in word for word in command.keywords):
            scored.append((2, 0, command))
    scored.sort(key=lambda row: (row[0], row[1], pool.index(row[2])))
    return tuple(command for _, _, command in scored)


def shortcuts_by_category(tab_labels: Optional[Sequence[str]] = None
                          ) -> Dict[str, List[Shortcut]]:
    """`tab_labels` is the strip currently on screen, so the ⌘1–⌘N note
    lists the tabs the reader can actually see rather than the equity
    labels — a panel that names a tab which is not on the page is worse
    than one that names none."""
    names = tuple(tab_labels) if tab_labels else MAIN_TABS
    grouped: Dict[str, List[Shortcut]] = {}
    for shortcut in SHORTCUTS:
        if shortcut.keys.startswith("⌘1"):
            shortcut = replace(
                shortcut,
                keys=f"⌘1 – ⌘{min(len(names), 9)}",
                note="In tab order: " + ", ".join(
                    f"{i + 1} {name}" for i, name in enumerate(names[:9])))
        grouped.setdefault(shortcut.category, []).append(shortcut)
    return grouped


def listener_html(enabled: bool = True, pending_tab: Optional[int] = None,
                  focus_panel: Optional[str] = None) -> str:
    """The key listener, as an invisible component iframe.

    Runs inside the iframe, binds on window.parent.document, and reaches
    Streamlit by clicking hidden buttons that finance.py renders.

    `focus_panel` is the st-key of a panel that has just opened. Both
    panels render where the script reaches them — near the top of the
    main column, which is a long way below the fold once the symbol
    header, quick stats and tutorial card are on screen. Measured live:
    the palette opened 995px below the viewport, so ⌘K looked like it had
    done nothing at all. The component scrolls it into view and focuses
    its input, which is also what makes a palette usable — you should be
    able to type the moment it opens.

    `pending_tab` is how the PALETTE changes tabs. st.tabs offers no way
    to select a tab from Python — there is no index= and no session_state
    key — so "Go to Risk & Technicals" cannot be honoured server-side at
    all. The palette instead parks the index in session_state, and the
    component clicks that tab once when it next mounts. Doing it on mount
    rather than on a timer means it happens in the same frame the palette
    closes, so it reads as one action.
    """
    config = {
        "triggers": TRIGGERS,
        "assetPrefix": ASSET_TRIGGER_PREFIX,
        # Only tabs[0] is read, to identify the main strip among the
        # page's nested tab groups — and "Overview" is the first tab for
        # every asset class, so this stays correct as labels change.
        "tabs": list(MAIN_TABS),
        "enabled": bool(enabled),
        "pendingTab": pending_tab if isinstance(pending_tab, int) else None,
        "focusPanel": focus_panel or None,
    }
    return """
<script>
(function () {
  const CONFIG = __CONFIG__;
  if (!CONFIG.enabled) return;

  let pdoc;
  try { pdoc = window.parent.document; } catch (err) { return; }
  if (!pdoc) return;

  // Every rerun mounts a FRESH iframe, and each one would add another
  // listener to the same parent document — after five reruns one
  // keypress would fire five times. The previous handler is removed
  // through a marker parked on the parent window.
  const SLOT = "__quantixKeyHandler";
  if (window.parent[SLOT]) {
    pdoc.removeEventListener("keydown", window.parent[SLOT], true);
  }

  function clickTrigger(name) {
    const key = CONFIG.triggers[name];
    if (!key) return false;
    const host = pdoc.querySelector('[class*="st-key-' + key + '"]');
    if (!host) return false;
    const button = host.querySelector("button");
    if (!button) return false;
    button.click();
    return true;
  }

  // The main strip is identified by its first tab's label. This page has
  // nested tab groups, so "the first .stTabs" would pick whichever
  // happened to render first.
  function mainTabList() {
    const lists = pdoc.querySelectorAll('[data-baseweb="tab-list"]');
    for (const list of lists) {
      const first = list.querySelector('[data-baseweb="tab"]');
      if (first && first.innerText.trim() === CONFIG.tabs[0]) return list;
    }
    return null;
  }

  function goToTab(index) {
    const list = mainTabList();
    if (!list) return false;
    const tabs = list.querySelectorAll('[data-baseweb="tab"]');
    if (index >= tabs.length) return false;
    tabs[index].click();
    return true;
  }

  function isTyping(target) {
    if (!target) return false;
    const tag = (target.tagName || "").toLowerCase();
    return tag === "input" || tag === "textarea" || tag === "select"
        || target.isContentEditable === true;
  }

  function clickAsset(index) {
    const host = pdoc.querySelector(
      '[class*="st-key-' + CONFIG.assetPrefix + index + '"]');
    if (!host) return false;
    const button = host.querySelector("button");
    if (!button) return false;
    button.click();
    return true;
  }

  const handler = function (event) {
    const mod = event.metaKey || event.ctrlKey;
    const typing = isTyping(event.target);

    // Asset class: Alt+1..6. Matched on event.code, NOT event.key —
    // macOS composes Option+1 into "\u00a1", so a key-based match would
    // fire on Windows and never on a Mac. Probed on the running page:
    // the event arrives with altKey true and is not defaultPrevented.
    if (event.altKey && !mod && !event.shiftKey) {
      const m = /^Digit([1-9])$/.exec(event.code || "");
      if (m && !typing) {
        if (clickAsset(parseInt(m[1], 10) - 1)) event.preventDefault();
        return;
      }
    }

    // Bare keys yield to typing; "?" is a character someone enters into
    // the ticker box, and stealing it would be a bug.
    if (!mod) {
      if (typing) return;
      if (event.key === "?") { event.preventDefault(); clickTrigger("shortcuts"); }
      else if (event.key === "Escape") { clickTrigger("close"); }
      return;
    }

    if (event.shiftKey && event.key.toLowerCase() === "a") {
      event.preventDefault(); clickTrigger("new_alert"); return;
    }
    if (event.shiftKey) return;

    const key = event.key.toLowerCase();
    if (key === "k") { event.preventDefault(); clickTrigger("palette"); return; }
    if (key === "/") { event.preventDefault(); clickTrigger("help"); return; }
    if (key >= "1" && key <= "9") {
      // Only claim the key if the strip is actually on screen; otherwise
      // leave the browser's own binding alone rather than swallowing it.
      if (goToTab(parseInt(key, 10) - 1)) event.preventDefault();
    }
  };

  pdoc.addEventListener("keydown", handler, true);
  window.parent[SLOT] = handler;

  // Both of the actions below have to wait for the parent to paint.
  //
  // This iframe mounts near the TOP of the script, while the tab strip is
  // created hundreds of lines later after every fetch and computation
  // this page does — measured at roughly ten seconds on a cold load. A
  // short retry loop expired long before the tabs existed, so the
  // palette's "Go to Risk & Technicals" closed the palette and then did
  // nothing at all. A MutationObserver fires the moment the element
  // appears instead of guessing at an interval, with a generous ceiling
  // so nothing observes forever.
  const WAIT_MS = 30000;
  function whenReady(find, act) {
    const found = find();
    if (found) { act(found); return; }
    let done = false;
    const observer = new MutationObserver(function () {
      if (done) return;
      const hit = find();
      if (hit) { done = true; observer.disconnect(); act(hit); }
    });
    observer.observe(pdoc.body, { childList: true, subtree: true });
    setTimeout(function () {
      if (!done) { done = true; observer.disconnect(); }
    }, WAIT_MS);
  }

  // A panel that has just opened, brought to where the reader is.
  if (CONFIG.focusPanel) {
    whenReady(
      function () { return pdoc.querySelector('[class*="st-key-' + CONFIG.focusPanel + '"]'); },
      function (panel) {
        panel.scrollIntoView({ block: "center", behavior: "smooth" });
        const field = panel.querySelector("input");
        if (field) field.focus();
      });
  }

  // A tab the palette asked for.
  //
  // Parked on the PARENT window rather than acted on directly, because
  // this iframe does not live long enough. Streamlit replaces the
  // component on the next rerun — and this page reruns constantly, not
  // least from its own polling fragments — so the iframe that carried
  // the request was torn down, observer and all, before the tab strip
  // appeared hundreds of lines further down the script. Verified live:
  // the config arrived with pendingTab 4 and nothing happened, while
  // clicking the same tab by hand worked. The parent survives, so the
  // request waits there and whichever iframe is alive when the strip
  // lands performs it, then clears it so it fires exactly once.
  const PENDING = "__quantixPendingTab";
  if (CONFIG.pendingTab !== null && CONFIG.pendingTab !== undefined) {
    window.parent[PENDING] = CONFIG.pendingTab;
  }
  const pending = window.parent[PENDING];
  if (pending !== null && pending !== undefined) {
    // Clicking once is not enough. st.tabs holds no state, so every
    // rerun rebuilds the strip with the FIRST tab selected — and closing
    // the palette IS a rerun, with more following while the page
    // finishes loading. Measured live: the click landed and the flag
    // cleared, and the strip was back on Overview a moment later.
    //
    // So the request is re-applied until the target actually holds, and
    // only then cleared. It also stops the moment the reader picks a
    // different tab themselves — a request that kept yanking them back
    // would be far worse than one that quietly gave up.
    let settled = 0;
    const deadline = Date.now() + WAIT_MS;
    const enforce = setInterval(function () {
      const list = mainTabList();
      if (!list) { if (Date.now() > deadline) clearInterval(enforce); return; }
      const tabs = list.querySelectorAll('[data-baseweb="tab"]');
      if (pending >= tabs.length || Date.now() > deadline) {
        clearInterval(enforce); window.parent[PENDING] = null; return;
      }
      const selected = [...tabs].findIndex(
        function (t) { return t.getAttribute("aria-selected") === "true"; });
      if (selected === pending) {
        // Hold for two consecutive checks before trusting it.
        if (++settled >= 2) { clearInterval(enforce); window.parent[PENDING] = null; }
        return;
      }
      if (selected > 0 && selected !== pending && settled > 0) {
        // The reader moved somewhere else on purpose. Stand down.
        clearInterval(enforce); window.parent[PENDING] = null; return;
      }
      tabs[pending].click();
    }, 400);
  }
})();
</script>
""".replace("__CONFIG__", json.dumps(config))
