# GitHub Pages deployment (MkDocs)

This repository uses MkDocs for documentation. A GitHub Actions workflow is included to build the site and deploy it to GitHub Pages automatically on pushes to `main` and `dev`.

What I added
- `.github/workflows/pages.yml` — builds MkDocs and publishes the generated `site/` directory via GitHub Pages Actions.

How it works
- On push to `main` or `dev`, the workflow installs MkDocs and plugins, runs `mkdocs build --site-dir site`, and uses the official Pages actions to publish the `site/` directory.

Verify Pages is published
1. Go to your repository on GitHub → Settings → Pages. The Pages source will be handled by the Actions workflow and should show a deployment after the workflow runs.
2. Open the published URL shown in the Pages settings or the Actions workflow run summary.

- Notes & recommendations
- The repository is set to use **GitHub Pages (free for public/open-source repos)**. I updated `mkdocs.yml` to use the Pages URL:

	`https://jimmyjdejesus-cmyk.github.io/allele-ai/`

	This URL is free and automatically served by GitHub. You do not need a custom domain to publish.
- If you have a custom domain, add a `CNAME` file with your domain name either to the repository root (preferred) or ensure it is present in the built `site/` artifact before deployment. Example CNAME contents:

```
docs.yourdomain.com
```

- Ensure repository `Settings → Pages` shows the `github-pages` environment or the Pages deployment — the workflow will create deployments automatically after a successful job.

Security/permissions
- The workflow requests the `pages: write` permission (minimum required) and uses `actions/configure-pages` + `actions/upload-pages-artifact` + `actions/deploy-pages` which are GitHub-maintained actions for Pages deployments.

If you'd like, I can:
- Add an example `CNAME` (if you provide the domain),
- Update `mkdocs.yml` to set `site_url` to the Pages URL, and
- Add a short verification script or Action that runs after deploy and alerts if the site is unreachable.
