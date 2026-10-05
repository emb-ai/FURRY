import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { parseArgs } from "node:util";
import { lint } from "markdownlint/sync";
import { PAGES, renderWiki, validateKnowledgeBase, validateLinks } from "./docs-lib.mjs";

function git(root, ...args) {
  return execFileSync("git", ["-C", root, ...args], { encoding: "utf8" }).trimEnd();
}

function within(parent, child) {
  const relative = path.relative(parent, child);
  return relative === "" || (!relative.startsWith(`..${path.sep}`) && relative !== ".." && !path.isAbsolute(relative));
}

function markdownFiles(root) {
  const names = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "*.md")
    .split("\0").filter(Boolean);
  return new Map([...new Set(names)].map((name) => [name, fs.readFileSync(path.join(root, name), "utf8")]));
}

function check(root) {
  const files = markdownFiles(root);
  const errors = [
    ...validateLinks(files, (name) => fs.existsSync(path.join(root, name))),
    ...validateKnowledgeBase(files),
  ];
  if (errors.length) throw new Error(errors.join("\n"));
  console.log(`Checked ${files.size} Markdown files, internal links, page navigation and 12 bibliography entries.`);
}

function canonicalPath(requested) {
  let parent = requested;
  const suffix = [];
  while (!fs.existsSync(parent)) {
    suffix.unshift(path.basename(parent));
    parent = path.dirname(parent);
  }
  return path.join(fs.realpathSync(parent), ...suffix);
}

function wiki(root, options) {
  const modes = [options.write, options.check, options["dry-run"]].filter(Boolean);
  if (modes.length > 1) throw new Error("Choose one of --dry-run, --write or --check");
  if (!options.output) throw new Error("Provide --output pointing to a separate FURRY Wiki checkout");
  const sha = git(root, "rev-parse", "--verify", "--end-of-options", `${options.ref ?? "HEAD"}^{commit}`);
  const files = new Map(PAGES.map((page) => [page.source, git(root, "show", `${sha}:${page.source}`) + "\n"]));
  const rendered = renderWiki(files, sha);
  const requested = path.resolve(options.output);
  const output = canonicalPath(requested);
  const source = fs.realpathSync(root);
  if (within(source, output) || within(output, source)) {
    throw new Error("Wiki output must be separate from the source repository, not its ancestor or descendant");
  }
  if (options.write) {
    if (!fs.existsSync(output)) throw new Error("Clone the initialized Wiki first; --write does not create a checkout");
    if (fs.realpathSync(git(output, "rev-parse", "--show-toplevel")) !== output) {
      throw new Error("Output must be the root of a separate Wiki checkout");
    }
    const origin = git(output, "remote", "get-url", "origin").replace(/\.git$/, "");
    if (!["https://github.com/emb-ai/FURRY.wiki", "git@github.com:emb-ai/FURRY.wiki"].includes(origin)) {
      throw new Error("Output origin is not emb-ai/FURRY.wiki");
    }
    if (git(output, "status", "--porcelain", "--untracked-files=all")) {
      throw new Error("Wiki checkout must be clean before writing; preserve or commit existing changes first");
    }
  }
  const changes = [];
  for (const [name, text] of rendered) {
    const target = path.join(output, name);
    if (fs.lstatSync(target, { throwIfNoEntry: false })?.isSymbolicLink()) {
      throw new Error(`Refusing symlink target: ${name}`);
    }
    if (!fs.existsSync(target) || fs.readFileSync(target, "utf8") !== text) changes.push(name);
  }
  console.log(`Source commit: ${sha} (committed files only)`);
  console.log(`${options.write ? "Write" : options.check ? "Check" : "Dry run"}: ${changes.length}/${rendered.size} generated files differ in ${output}`);
  for (const name of changes) console.log(`  ${name}`);
  if (options.check && changes.length) throw new Error("Wiki differs from the selected source commit");
  if (options.write) {
    for (const name of changes) fs.writeFileSync(path.join(output, name), rendered.get(name));
    console.log("Generated files updated. Review the diff; nothing was committed or pushed.");
  }
}

try {
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options: {
      output: { type: "string" }, ref: { type: "string" },
      write: { type: "boolean" }, check: { type: "boolean" }, "dry-run": { type: "boolean" },
    },
  });
  const root = git(process.cwd(), "rev-parse", "--show-toplevel");
  if (positionals.length !== 1) throw new Error("Usage: docs.mjs lint | check | wiki --output PATH [--ref COMMIT] [--dry-run|--write|--check]");
  if (["lint", "check"].includes(positionals[0])) {
    if (Object.keys(values).length) throw new Error("The lint and check commands take no flags");
    if (positionals[0] === "lint") {
      const files = markdownFiles(root);
      const results = lint({ strings: Object.fromEntries(files),
        config: JSON.parse(fs.readFileSync(path.join(root, ".markdownlint.json"), "utf8")) });
      const diagnostics = Object.entries(results).flatMap(([name, errors]) =>
        errors.map((error) => `${name}:${error.lineNumber}: ${error.ruleNames[0]} ${error.ruleDescription}${error.errorDetail ? ` (${error.errorDetail})` : ""}`))
        .join("\n");
      if (diagnostics) throw new Error(diagnostics);
      console.log(`Markdown formatting passed for ${files.size} files.`);
    } else {
      check(root);
    }
  } else if (positionals[0] === "wiki") {
    wiki(root, values);
  } else {
    throw new Error(`Unknown command: ${positionals[0]}`);
  }
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}
