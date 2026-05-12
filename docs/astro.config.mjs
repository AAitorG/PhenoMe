// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

// GitHub Pages project site: https://<user>.github.io/<repo>/
const site = "https://AAitorG.github.io";
const base = "/PhenoMe/";

const TUTORIALS_URL = "https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials";
const EXAMPLES_URL = "https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/examples";

export default defineConfig({
  site,
  base,
  publicDir: "./assets",
  trailingSlash: "always",
  redirects: {
    [`${base}guides/property-reference/`]: `${base}concepts/property-interpretation/`,
    [`${base}guides/interpretability/`]: `${base}concepts/embeddings-interpretability/`,
    [`${base}concepts/image-properties-and-presets/`]: `${base}concepts/property-interpretation/`,
    [`${base}concepts/interpretability/`]: `${base}concepts/embeddings-interpretability/`,
    [`${base}guides/custom-properties/`]: `${base}guides/select-properties/`,
    [`${base}guides/custom-properties/presets/`]: `${base}guides/select-properties/presets/`,
    [`${base}guides/custom-properties/writing-functions/`]: `${base}guides/select-properties/writing-functions/`,
    [`${base}guides/custom-properties/advanced-recipes/`]: `${base}guides/select-properties/advanced-recipes/`,
  },
  integrations: [
    starlight({
      title: "PhenoMe",
      description:
        "Dataset-agnostic phenotyping with deep learning embeddings (Python package: phenome).",
      logo: {
        src: "./src/assets/logo_horizontal.png",
        alt: "PhenoMe",
        replacesTitle: true,
      },
      favicon: "/logo_minimal.png",
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/AAitorG/PhenoMe",
        },
      ],
      customCss: ["./src/styles/custom.css"],
      sidebar: [
        { label: "Home", link: "/" },
        {
          label: "Start here",
          items: [
            { label: "Getting started", link: "/getting-started/" },
            { label: "Learning paths", link: "/user-paths/" },
            { label: "Common workflows", link: "/workflows/" },
          ],
        },
        {
          label: "Concepts",
          collapsed: true,
          items: [
            { label: "Overview", link: "/concepts/" },
            { label: "Embeddings", link: "/concepts/embeddings/" },
            { label: "Channel modes", link: "/concepts/channel-modes/" },
            { label: "Distance metrics", link: "/concepts/distances/" },
            { label: "Metadata & properties", link: "/concepts/metadata-and-properties/" },
            { label: "Property interpretation", link: "/concepts/property-interpretation/" },
            { label: "Visualization & results", link: "/concepts/visualization-and-results/" },
            { label: "Embeddings interpretability", link: "/concepts/embeddings-interpretability/" },
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
              label: "Select properties",
              collapsed: true,
              items: [
                { label: "Overview", link: "/guides/select-properties/" },
                { label: "Presets", link: "/guides/select-properties/presets/" },
                { label: "Writing functions", link: "/guides/select-properties/writing-functions/" },
                { label: "Advanced recipes", link: "/guides/select-properties/advanced-recipes/" },
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
          label: "Notebooks",
          collapsed: true,
          items: [
            {
              label: "Tutorials",
              collapsed: false,
              items: [
                { label: "01 Interactive quickstart", link: `${TUTORIALS_URL}/01_interactive_quickstart.ipynb`, attrs: { target: "_blank", rel: "noopener" } },
                { label: "02 Inspect your images", link: `${TUTORIALS_URL}/02_inspect_your_images.ipynb`, attrs: { target: "_blank", rel: "noopener" } },
                { label: "03 Core phenotyping workflow", link: `${TUTORIALS_URL}/03_core_phenotyping_workflow.ipynb`, attrs: { target: "_blank", rel: "noopener" } },
                { label: "04 Exploratory & explainability", link: `${TUTORIALS_URL}/04_exploratory_analysis_and_explainability.ipynb`, attrs: { target: "_blank", rel: "noopener" } },
                { label: "05 Extending with plugins", link: `${TUTORIALS_URL}/05_extending_phenome_plugins.ipynb`, attrs: { target: "_blank", rel: "noopener" } },
                { label: "06 Custom model and preprocessing", link: `${TUTORIALS_URL}/06_custom_model_and_preprocessing.ipynb`, attrs: { target: "_blank", rel: "noopener" } },
              ],
            },
            {
              label: "Examples",
              collapsed: false,
              items: [
                { label: "BBBC014 Two cell lines", link: `${EXAMPLES_URL}/bbbc014_two_cell_lines.ipynb`, attrs: { target: "_blank", rel: "noopener" } },
                { label: "BBBC021 Compound screen", link: `${EXAMPLES_URL}/bbbc021_compound_screen.ipynb`, attrs: { target: "_blank", rel: "noopener" } },
                { label: "CellCognition Time-lapse", link: `${EXAMPLES_URL}/cellcognition_time_lapse.ipynb`, attrs: { target: "_blank", rel: "noopener" } },
                { label: "E. coli Drug timecourse", link: `${EXAMPLES_URL}/ecoli_drug_timecourse.ipynb`, attrs: { target: "_blank", rel: "noopener" } },
                { label: "HPA Subcellular phenotypes", link: `${EXAMPLES_URL}/hpa_subcellular_phenotypes.ipynb`, attrs: { target: "_blank", rel: "noopener" } },
              ],
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
            { label: "Custom model wrappers", link: "/guides/custom-model-wrapper/" },
            { label: "External checkpoints", link: "/guides/external-checkpoints/" },
            { label: "Developer guide", link: "/guides/developer-guide/" },
            { label: "Testing & quality", link: "/guides/testing/" },
            { label: "Contributing to docs", link: "/guides/contributing-guide/" },
          ],
        },
        {
          label: "More",
          items: [
            { label: "Glossary", link: "/glossary/" },
            { label: "FAQ", link: "/faq/" },
          ],
        },
      ],
    }),
  ],
});
