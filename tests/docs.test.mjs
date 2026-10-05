import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import { PAGES, REPO, headings, links, parse, renderWiki, repositoryLink, validateLinks } from "../tools/docs-lib.mjs";

const SHA = "a".repeat(40);
const CLI = fileURLToPath(new URL("../tools/docs.mjs", import.meta.url));

function pages() {
  return new Map(PAGES.map((page) => [page.source, `# ${page.title}\n\nA source page.\n`]));
}

function git(cwd, ...args) {
  return execFileSync("git", ["-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", ...args],
    { cwd, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }).trim();
}

function commit(cwd) {
  git(cwd, "add", "--all");
  git(cwd, "commit", "-m", "test fixture");
}

function fixture(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "furry-docs-"));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const source = path.join(dir, "source");
  const wiki = path.join(dir, "wiki");
  for (const folder of [source, wiki]) {
    fs.mkdirSync(folder);
    git(folder, "init", "-b", "main");
    git(folder, "config", "user.name", "Documentation Test");
    git(folder, "config", "user.email", "docs-test@example.invalid");
  }
  fs.mkdirSync(path.join(source, "docs"));
  for (const [name, text] of pages()) fs.writeFileSync(path.join(source, name), text);
  commit(source);
  fs.writeFileSync(path.join(wiki, "Home.md"), "# Initial Wiki placeholder\n");
  fs.writeFileSync(path.join(wiki, "Unrelated.md"), "Keep this page.\n");
  commit(wiki);
  git(wiki, "remote", "add", "origin", "https://github.com/emb-ai/FURRY.wiki.git");
  const run = (...args) => spawnSync(process.execPath, [CLI, "wiki", "--output", wiki, ...args],
    { cwd: source, encoding: "utf8" });
  return { dir, source, wiki, run };
}

test("Markdown parsing ignores example links in fenced and inline code", () => {
  const tree = parse("[real](Home.md)\n\n`[inline](missing.md)`\n\n```text\n[fenced](missing.md)\n```\n");
  assert.equal(links(tree).length, 1);
  assert.equal(links(tree)[0].url, "Home.md");
});

test("heading IDs follow GitHub text normalization and duplicate numbering", () => {
  assert.deepEqual([...headings(parse("# Hello `world`\n\n## Hello world\n"))],
    ["hello-world", "hello-world-1"]);
});

test("internal links validate decoded paths and heading anchors", () => {
  const files = new Map([
    ["docs/A.md", "# A\n\n[File](B%20file.md#hello-world)\n\n[license](../LICENSE)\n"],
    ["docs/B file.md", "# Hello world\n"],
  ]);
  assert.deepEqual(validateLinks(files, (p) => p === "LICENSE"), []);
});

test("missing files and anchors are diagnosed separately", () => {
  const files = new Map([["docs/A.md", "# A\n\n[bad](missing.md)\n\n[anchor](#nope)\n"]]);
  const errors = validateLinks(files, () => false);
  assert.equal(errors.length, 2);
  assert.match(errors[0], /missing target/);
  assert.match(errors[1], /missing anchor/);
});

test("unsafe protocols and links outside the repository are rejected", () => {
  assert.throws(() => repositoryLink("docs/A.md", "../../secret"), /escapes/);
  assert.throws(() => repositoryLink("docs/A.md", "javascript:alert(1)"), /protocol/);
  assert.equal(repositoryLink("docs/A.md", "https://example.org/"), null);
});

test("Wiki conversion rewrites table/reference links but preserves code and local anchors", () => {
  const files = pages();
  files.set("docs/Home.md", "# Home\n\n[local](#home)\n\n[guide][g]\n\n[g]: Getting-Started.md#next\n\n| Item | Source |\n| --- | --- |\n| Doc | [Read](Common-Ground.md) |\n\n[license](../LICENSE)\n\n[external](https://example.org/)\n\n```text\n[example](Common-Ground.md)\n```\n");
  const output = renderWiki(files, SHA);
  const home = output.get("Home.md");
  assert.match(home, new RegExp(`${REPO}/wiki/Getting-Started#next`));
  assert.match(home, new RegExp(`${REPO}/wiki/Common-Ground`));
  assert.match(home, new RegExp(`${REPO}/blob/${SHA}/LICENSE`));
  assert.ok(home.includes("[example](Common-Ground.md)"));
  assert.ok(home.includes("[local](#home)"));
  assert.ok(home.includes("https://example.org/"));
  assert.equal(output.size, 10);
  for (const page of PAGES) assert.ok(output.get("_Sidebar.md").includes(`/wiki/${page.slug}`));
  assert.ok(output.get("_Footer.md").includes(SHA));
  assert.deepEqual(renderWiki(files, SHA), output);
});

test("a full immutable SHA and all eight pages are required", () => {
  assert.throws(() => renderWiki(pages(), "main"), /immutable/);
  const files = pages();
  files.delete("docs/Home.md");
  assert.throws(() => renderWiki(files, SHA), /no docs\/Home.md/);
});

test("lint succeeds for clean Markdown and reports file, line and rule on failure", (t) => {
  const f = fixture(t);
  fs.writeFileSync(path.join(f.source, ".markdownlint.json"), "{}\n");
  const run = () => spawnSync(process.execPath, [CLI, "lint"], { cwd: f.source, encoding: "utf8" });
  const clean = run();
  assert.equal(clean.status, 0, clean.stderr);
  assert.match(clean.stdout, /Markdown formatting passed/);
  fs.writeFileSync(path.join(f.source, "docs/Home.md"), "# Home\n\n##Skipped level\n");
  const invalid = run();
  assert.equal(invalid.status, 1);
  assert.match(invalid.stderr, /docs\/Home.md:3: MD018/);
});

test("default dry run neither creates directories nor changes the Wiki", (t) => {
  const f = fixture(t);
  const before = git(f.wiki, "status", "--porcelain");
  const result = f.run();
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /Dry run/);
  assert.equal(git(f.wiki, "status", "--porcelain"), before);
  const target = path.join(f.dir, "not-created");
  const missing = f.run("--output", target, "--dry-run");
  assert.equal(missing.status, 0, missing.stderr);
  assert.equal(fs.existsSync(target), false);
});

test("write uses committed source, preserves unrelated pages, and never commits", (t) => {
  const f = fixture(t);
  const before = git(f.wiki, "rev-parse", "HEAD");
  const sha = git(f.source, "rev-parse", "HEAD");
  fs.writeFileSync(path.join(f.source, "docs/Home.md"), "# Uncommitted secret draft\n");
  const result = f.run("--write");
  assert.equal(result.status, 0, result.stderr);
  const home = fs.readFileSync(path.join(f.wiki, "Home.md"), "utf8");
  assert.ok(home.includes(sha));
  assert.ok(!home.includes("Uncommitted secret draft"));
  assert.equal(fs.readFileSync(path.join(f.wiki, "Unrelated.md"), "utf8"), "Keep this page.\n");
  assert.equal(git(f.wiki, "rev-parse", "HEAD"), before);
  assert.notEqual(git(f.wiki, "status", "--porcelain"), "");
});

test("check detects drift, succeeds after rendering, and never writes", (t) => {
  const f = fixture(t);
  assert.equal(f.run("--check").status, 1);
  assert.equal(f.run("--write").status, 0);
  assert.equal(f.run("--check").status, 0);
  fs.appendFileSync(path.join(f.wiki, "Home.md"), "Manual drift\n");
  const result = f.run("--check");
  assert.equal(result.status, 1);
  assert.match(result.stderr, /differs/);
  assert.ok(fs.readFileSync(path.join(f.wiki, "Home.md"), "utf8").includes("Manual drift"));
});

test("repeat generation is unchanged after the Wiki changes are committed", (t) => {
  const f = fixture(t);
  assert.equal(f.run("--write").status, 0);
  commit(f.wiki);
  const result = f.run("--write");
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /0\/10 generated files differ/);
  assert.equal(git(f.wiki, "status", "--porcelain"), "");
});

test("write refuses dirty targets and wrong origins before changing files", (t) => {
  const f = fixture(t);
  fs.writeFileSync(path.join(f.wiki, "Untracked.txt"), "Keep me");
  assert.match(f.run("--write").stderr, /must be clean/);
  commit(f.wiki);
  git(f.wiki, "remote", "set-url", "origin", "https://github.com/other/project.wiki.git");
  assert.match(f.run("--write").stderr, /origin is not/);
  assert.equal(fs.readFileSync(path.join(f.wiki, "Home.md"), "utf8"), "# Initial Wiki placeholder\n");
});

test("write refuses a missing checkout and overlapping source/output paths", (t) => {
  const f = fixture(t);
  const missing = path.join(f.dir, "missing");
  assert.match(f.run("--output", missing, "--write").stderr, /Clone the initialized Wiki/);
  assert.equal(fs.existsSync(missing), false);
  for (const output of [f.source, path.join(f.source, "wiki"), f.dir]) {
    assert.match(f.run("--output", output).stderr, /must be separate/);
  }
});

test("dangling symlinks are rejected before any generated file is written", (t) => {
  const f = fixture(t);
  const external = path.join(f.dir, "must-not-exist");
  fs.symlinkSync(external, path.join(f.wiki, "_Sidebar.md"));
  commit(f.wiki);
  const result = f.run("--write");
  assert.equal(result.status, 1);
  assert.match(result.stderr, /symlink/);
  assert.equal(fs.existsSync(external), false);
  assert.equal(fs.readFileSync(path.join(f.wiki, "Home.md"), "utf8"), "# Initial Wiki placeholder\n");
});

test("conflicting modes are rejected without writes", (t) => {
  const f = fixture(t);
  assert.match(f.run("--write", "--dry-run").stderr, /Choose one/);
  assert.equal(git(f.wiki, "status", "--porcelain"), "");
});
