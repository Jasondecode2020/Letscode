# AGENTS

## Repository overview
This repository is a collection of LeetCode problem solutions, patterns, and study notes written in Markdown. It is not a typical software project with build or test scripts. Most work consists of creating, updating, and organizing Markdown content under:

- `template/` — pattern-style solution templates and examples
- `pattern/` — problem patterns and algorithms
- `list/` — curated problem lists
- `others/` — additional study notes and resources
- `task/` — user-created task files or workspace-specific notes

## What AI coding agents should do
- Treat the repository as documentation and learning material, not a codebase to compile.
- Preserve existing Markdown formatting, headings, and problem structure.
- When adding or updating problems, follow the repo's existing conventions for titles, numbered LeetCode entries, and code blocks.
- Prefer editing or extending existing `template`/`pattern` files rather than creating duplicate content.
- Keep changes focused and minimal unless the user explicitly requests broad restructuring.

## Key conventions
- Use Markdown headings (`#`, `##`, `###`) for section organization.
- Keep code samples in fenced code blocks with the appropriate language label, e.g. ````python````.
- Link to existing repository content when referencing related patterns or examples.
- Do not invent new build/test workflows for this repo; there is no evidence of a standard programming environment or CI configuration.

## When to create customization files
- Create or update `AGENTS.md` if the repository lacks a dedicated Copilot instruction file.
- If the repository later gains a `.github/copilot-instructions.md`, prefer updating that file instead.

## Notes for future work
- If asked to add new problem collections, create a new Markdown file in a relevant root directory and use existing naming conventions.
- If asked to organize material, keep the directory structure shallow and match the current `template/pattern/list` separation.
- Use the `README.md` as the primary project description if additional context is needed.
