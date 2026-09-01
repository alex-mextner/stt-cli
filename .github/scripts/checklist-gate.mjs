// Single source of truth for the PR checklist gate.
// Accessed via: imported by `workflow.yml` (github-script step) AND by
// `checklist-gate.test.mjs` (self-check). One parser, two callers — a green test proves
// the gate's logic without duplicating the regex.
//
// Assumptions: a "task-list item" is a Markdown bullet (`-`, `*`, or `+`) followed by
// ` [ ]` (unchecked) or ` [x]`/`[X]` (checked), optionally indented. Inline bracket pairs
// that are not task-list bullets are ignored.

// Line-anchored, optional leading whitespace, `-`/`*`/`+` bullet, single space, then
// exactly `[ ]` (unchecked). Checked boxes use `[x]`/`[X]`.
const UNCHECKED_RE = /^[ \t]*[-*+] \[ \]/;

/**
 * Return the list of unchecked task-list lines (trimmed) in a PR body.
 * Empty array means the gate passes (all checked, or no checkboxes at all).
 * Tolerates null/undefined/empty bodies (returns []).
 * @param {string | null | undefined} body
 * @returns {string[]}
 */
export function parseUnchecked(body) {
  if (!body) return [];
  return body
    .split(/\r?\n/)
    .filter((line) => UNCHECKED_RE.test(line))
    .map((line) => line.trim());
}
