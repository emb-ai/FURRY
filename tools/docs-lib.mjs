import path from "node:path";
import GithubSlugger from "github-slugger";
import { toString } from "mdast-util-to-string";
import remarkGfm from "remark-gfm";
import remarkParse from "remark-parse";
import remarkStringify from "remark-stringify";
import { unified } from "unified";
import { visit } from "unist-util-visit";

export const REPO = "https://github.com/emb-ai/FURRY";
export const PAGES = [
  ["Home", "Home"],
  ["Getting-Started", "Getting Started"],
  ["Literature-Review", "Literature Review"],
  ["TWIST2-Baseline", "TWIST2 Baseline"],
  ["Architecture-and-Decisions", "Architecture And Decisions"],
  ["Common-Ground", "Common Ground"],
  ["Data-and-Evaluation", "Data And Evaluation"],
  ["Contributing", "Contributing"],
].map(([slug, title]) => ({ slug, title, source: `docs/${slug}.md` }));

const processor = unified()
  .use(remarkParse)
  .use(remarkGfm)
  .use(remarkStringify, { bullet: "-", fences: true, listItemIndent: "one" });

export function parse(text) {
  return processor.parse(text);
}

export function links(tree) {
  const result = [];
  visit(tree, (node) => {
    if (["link", "image", "definition"].includes(node.type)) result.push(node);
  });
  return result;
}

export function headings(tree) {
  const slugger = new GithubSlugger();
  const result = new Set();
  visit(tree, "heading", (node) => result.add(slugger.slug(toString(node))));
  return result;
}

export function repositoryLink(source, url) {
  if (/^(?:[a-z][a-z0-9+.-]*:|\/\/)/i.test(url)) {
    if (!/^(?:https?:|mailto:|\/\/)/i.test(url)) {
      throw new Error(`Unsupported URL protocol: ${url}`);
    }
    return null;
  }
  const parsed = new URL(url, `https://furry.invalid/repository/${source}`);
  const prefix = "/repository/";
  if (!parsed.pathname.startsWith(prefix)) {
    throw new Error(`Link escapes the repository: ${url}`);
  }
  const target = path.posix.normalize(decodeURIComponent(parsed.pathname.slice(prefix.length)));
  if (target.startsWith("../") || path.posix.isAbsolute(target)) {
    throw new Error(`Link escapes the repository: ${url}`);
  }
  return {
    path: target,
    hash: parsed.hash,
    fragment: decodeURIComponent(parsed.hash.slice(1)),
    search: parsed.search,
  };
}

export function validateLinks(files, exists) {
  const errors = [];
  const trees = new Map([...files].map(([name, text]) => [name, parse(text)]));
  const anchors = new Map([...trees].map(([name, tree]) => [name, headings(tree)]));
  for (const [name, tree] of trees) {
    for (const node of links(tree)) {
      try {
        const target = repositoryLink(name, node.url);
        if (!target) continue;
        if (!files.has(target.path) && !exists(target.path)) {
          errors.push(`${name}: missing target ${node.url}`);
        } else if (target.fragment && anchors.has(target.path) &&
                   !anchors.get(target.path).has(target.fragment)) {
          errors.push(`${name}: missing anchor ${node.url}`);
        }
      } catch (error) {
        errors.push(`${name}: ${error.message}`);
      }
    }
  }
  return errors;
}

export function validateKnowledgeBase(files) {
  const errors = PAGES.filter((p) => !files.has(p.source)).map((p) => `Missing page: ${p.source}`);
  const home = parse(files.get("docs/Home.md") ?? "");
  const navigation = new Set(links(home).map((node) => repositoryLink("docs/Home.md", node.url)?.path));
  for (const page of PAGES.slice(1)) {
    if (!navigation.has(page.source)) errors.push(`Home does not link to ${page.source}`);
  }
  const literature = headings(parse(files.get("docs/Literature-Review.md") ?? ""));
  for (const id of ["twist-2025", "twist2-2025", "gmr-2025", "omnih2o-2024", "hover-2024",
    "sonic-2025", "avatarposer-2022", "questsim-2022", "xrobotoolkit-2025",
    "open-television-2024", "open-teach-2024", "anyteleop-2023"]) {
    if (!literature.has(id)) errors.push(`Missing bibliography entry: ${id}`);
  }
  return errors;
}

function encodePath(value) {
  return value.split("/").map(encodeURIComponent).join("/");
}

export function renderWiki(files, sha) {
  if (!/^[a-f0-9]{40}$/.test(sha)) throw new Error("An immutable full commit SHA is required");
  const pages = new Map(PAGES.map((page) => [page.source, page]));
  const result = new Map();
  for (const page of PAGES) {
    if (!files.has(page.source)) throw new Error(`Commit has no ${page.source}`);
    const tree = parse(files.get(page.source));
    for (const node of links(tree)) {
      const target = repositoryLink(page.source, node.url);
      if (!target || node.url.startsWith("#")) continue;
      const wikiPage = pages.get(target.path);
      const base = wikiPage
        ? `${REPO}/wiki/${wikiPage.slug}`
        : `${REPO}/blob/${sha}/${encodePath(target.path)}`;
      node.url = `${base}${target.search}${target.hash}`;
    }
    const banner = `<!-- furry-wiki-mirror -->\n\n> Generated from [${page.source}](${REPO}/blob/${sha}/${page.source}) at [${sha}](${REPO}/commit/${sha}).\n> Edit repository docs, not this mirror. This snapshot does not imply a merge to main.\n\n`;
    result.set(`${page.slug}.md`, banner + processor.stringify(tree));
  }
  result.set("_Sidebar.md", "<!-- furry-wiki-mirror -->\n\n" +
    PAGES.map((p) => `- [${p.title}](${REPO}/wiki/${p.slug})`).join("\n") + "\n");
  result.set("_Footer.md", `<!-- furry-wiki-mirror -->\n\n[Source commit ${sha}](${REPO}/commit/${sha}) | [Roadmap](${REPO}/issues/1) | [Edit the source docs](${REPO}/blob/${sha}/docs/Contributing.md)\n`);
  return result;
}
