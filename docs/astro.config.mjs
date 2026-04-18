// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

// GitHub Pages project site: https://<user>.github.io/<repo>/
const site = "https://AAitorG.github.io";
const base = "/PhenoMe/";

export default defineConfig({
  site,
  base,
  trailingSlash: "always",
  integrations: [
    starlight({
      title: "PhenoMe",
      description:
        "Dataset-agnostic phenotyping with deep learning embeddings (Python package: phenome).",
      logo: {
        src: "./src/assets/Logo.png",
        alt: "PhenoMe",
        replacesTitle: true,
      },
      favicon: "/favicon.png",
      // TODO: when bumping @astrojs/starlight to >= 0.32, migrate this object to
      // the array form: [{ icon: 'github', label: 'GitHub', href: '...' }].
      social: {
        github: "https://github.com/AAitorG/PhenoMe",
      },
      editLink: {
        baseUrl: "https://github.com/AAitorG/PhenoMe/edit/main/docs/src/content/docs/",
      },
      customCss: ["./src/styles/custom.css"],
      sidebar: [
        { label: "Home", link: "/" },
        {
          label: "Start here",
          items: [
            { label: "Getting started", link: "/getting-started/" },
            { label: "Quick start card", link: "/quick-start-card/" },
            { label: "Learning paths", link: "/user-paths/" },
            { label: "Core concepts", link: "/concepts/" },
            { label: "Workflows", link: "/workflows/" },
          ],
        },
        { label: "Guides", autogenerate: { directory: "guides" } },
        { label: "Advanced", autogenerate: { directory: "advanced" } },
        { label: "Examples", autogenerate: { directory: "examples" } },
        {
          label: "More",
          items: [
            { label: "FAQ", link: "/faq/" },
            { label: "Glossary", link: "/glossary/" },
            { label: "Quick reference", link: "/quick-reference/" },
            { label: "Function location", link: "/function-location-guide/" },
          ],
        },
      ],
    }),
  ],
});
