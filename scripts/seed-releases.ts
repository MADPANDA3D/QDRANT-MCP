import { releases } from "./release-data";

const repoEnv = process.env.GITHUB_RELEASES_REPO ?? process.env.GITHUB_REPOSITORY;
const token = process.env.GITHUB_TOKEN;

if (!repoEnv) {
  console.error("Missing GITHUB_RELEASES_REPO or GITHUB_REPOSITORY.");
  process.exit(1);
}

if (!token) {
  console.error("Missing GITHUB_TOKEN.");
  process.exit(1);
}

const [owner, repo] = repoEnv.split("/");

if (!owner || !repo) {
  console.error(`Invalid repository value: ${repoEnv}`);
  process.exit(1);
}

const apiBase = `https://api.github.com/repos/${owner}/${repo}`;

const compareVersions = (a: string, b: string) => {
  const toParts = (value: string) =>
    value
      .replace(/^v/i, "")
      .split(".")
      .map((part) => Number(part));
  const aParts = toParts(a);
  const bParts = toParts(b);
  for (let i = 0; i < Math.max(aParts.length, bParts.length); i += 1) {
    const diff = (aParts[i] ?? 0) - (bParts[i] ?? 0);
    if (diff !== 0) return diff;
  }
  return 0;
};

const formatBody = (summary: string, highlights: string[], date: string) => {
  const bulletList = highlights.map((item) => `- ${item}`).join("\n");
  return `## Summary\n${summary}\n\n## Highlights\n${bulletList}\n\n_Release date: ${date}_`;
};

const request = async (path: string, options?: RequestInit) => {
  const response = await fetch(`${apiBase}${path}`, {
    ...options,
    headers: {
      Accept: "application/vnd.github+json",
      Authorization: `Bearer ${token}`,
      ...(options?.headers ?? {}),
    },
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${text}`);
  }

  return response.json();
};

const ensureRelease = async () => {
  const ordered = [...releases].sort((a, b) => compareVersions(a.version, b.version));

  for (const release of ordered) {
    const tag = `v${release.version}`;
    try {
      await request(`/releases/tags/${tag}`);
      console.log(`Release ${tag} already exists, skipping.`);
      continue;
    } catch (error) {
      const message = (error as Error).message;
      if (!message.includes("404")) {
        throw error;
      }
    }

    console.log(`Creating release ${tag}...`);
    await request("/releases", {
      method: "POST",
      body: JSON.stringify({
        tag_name: tag,
        name: release.title,
        body: formatBody(release.summary, release.highlights, release.date),
        target_commitish: "main",
        draft: false,
        prerelease: false,
      }),
    });
  }
};

ensureRelease()
  .then(() => {
    console.log("Release seeding complete.");
  })
  .catch((error) => {
    console.error("Release seeding failed:", error);
    process.exit(1);
  });
