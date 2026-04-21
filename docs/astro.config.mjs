// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

// GitHub Pages project site: https://<user>.github.io/<repo>/
const site = "https://AAitorG.github.io";
const base = "/PhenoMe/";

const NOTEBOOKS = "https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials";

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
      // Starlight 0.31 accepts the object form below. When upgrading to
      // >= 0.32, migrate to the array form: [{ icon: 'github', label: 'GitHub', href: '...' }].
      social: {
        github: "https://github.com/AAitorG/PhenoMe",
      },
      customCss: ["./src/styles/custom.css"],
      sidebar: [
        { label: "Home", link: "/" },
        {
          label: "Start here",
          items: [
            { label: "Getting started", link: "/getting-started/" },
            { label: "Learning paths", link: "/user-paths/" },
            {
              label: "Core concepts",
              collapsed: true,
              items: [
                { label: "Overview", link: "/concepts/" },
                { label: "Embeddings", link: "/concepts/embeddings/" },
                { label: "Channel modes", link: "/concepts/channel-modes/" },
                { label: "Distance metrics", link: "/concepts/distances/" },
                { label: "Metadata & properties", link: "/concepts/metadata-and-properties/" },
                { label: "Visualization & results", link: "/concepts/visualization-and-results/" },
              ],
            },
            { label: "Common workflows", link: "/workflows/" },
          ],
        },
        {
          label: "For your data",
          items: [
            { label: "Data setup", link: "/guides/data-setup/" },
            {
              label: "Experiment details (metadata)",
              collapsed: true,
              items: [
                { label: "Overview", link: "/guides/experiment-details/" },
                { label: "Path templates", link: "/guides/experiment-details/path-templates/" },
                { label: "CSV lookup", link: "/guides/experiment-details/csv-lookup/" },
                { label: "Mask discovery", link: "/guides/experiment-details/mask-discovery/" },
                { label: "Advanced patterns", link: "/guides/experiment-details/advanced-patterns/" },
              ],
            },
            {
              label: "Custom properties",
              collapsed: true,
              items: [
                { label: "Overview", link: "/guides/custom-properties/" },
                { label: "Presets", link: "/guides/custom-properties/presets/" },
                { label: "Writing functions", link: "/guides/custom-properties/writing-functions/" },
                { label: "Advanced recipes", link: "/guides/custom-properties/advanced-recipes/" },
              ],
            },
            {
              label: "Best practices",
              collapsed: true,
              autogenerate: { directory: "guides/best-practices" },
            },
          ],
        },
        {
          label: "Tutorials (notebooks)",
          collapsed: true,
          items: [
            { label: "01 Interactive quickstart", link: `${NOTEBOOKS}/01_interactive_quickstart.ipynb`, attrs: { target: "_blank", rel: "noopener" } },
            { label: "02 Inspect your images", link: `${NOTEBOOKS}/02_inspect_your_images.ipynb`, attrs: { target: "_blank", rel: "noopener" } },
            { label: "03 Core phenotyping workflow", link: `${NOTEBOOKS}/03_core_phenotyping_workflow.ipynb`, attrs: { target: "_blank", rel: "noopener" } },
            { label: "04 Advanced metadata handling", link: `${NOTEBOOKS}/04_advanced_metadata_handling.ipynb`, attrs: { target: "_blank", rel: "noopener" } },
            { label: "05 Exploratory & explainability", link: `${NOTEBOOKS}/05_exploratory_analysis_and_explainability.ipynb`, attrs: { target: "_blank", rel: "noopener" } },
            { label: "06 Extending with plugins", link: `${NOTEBOOKS}/06_extending_phenome_plugins.ipynb`, attrs: { target: "_blank", rel: "noopener" } },
            {
              label: "Examples",
              collapsed: true,
              autogenerate: { directory: "examples" },
            },
          ],
        },
        {
          label: "API & data formats",
          collapsed: true,
          items: [
            { label: "Overview", link: "/advanced/" },
            { label: "Function location", link: "/function-location-guide/" },
            {
              label: "API",
              collapsed: true,
              autogenerate: { directory: "advanced/api" },
            },
            { label: "HDF5 database protocol", link: "/advanced/database_protocol/" },
          ],
        },
        {
          label: "For developers",
          collapsed: true,
          items: [
            { label: "Architecture", link: "/guides/architecture/" },
            { label: "Extending the pipeline", link: "/guides/extending/" },
            { label: "Plugins", link: "/guides/plugins/" },
            { label: "External checkpoints", link: "/guides/external-checkpoints/" },
            { label: "Developer guide", link: "/guides/developer-guide/" },
            { label: "Testing & quality", link: "/guides/testing/" },
            { label: "Contributing to docs", link: "/guides/contributing-guide/" },
          ],
        },
        {
          label: "More",
          items: [
            { label: "Property reference", link: "/guides/property-reference/" },
            { label: "Interpretability", link: "/guides/interpretability/" },
            { label: "Glossary", link: "/glossary/" },
            { label: "FAQ", link: "/faq/" },
          ],
        },
      ],
    }),
  ],
});
