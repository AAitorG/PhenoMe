
Documentation:
In general the documentation feels a little bit overwhelming and hard to navigate. I didn’t check all the documentation files in depth, but here are the comments from the one I have checked. I would make more focus on two documentations: noon expert users and developers.
For users:
Installation
How to set up your data
Example notebooks
For developers:
Here I wouldn’t make a guided documentation, instead a main page that would point to the other documentation files but with some clear explanation about what to expect. The documentation of the API, I’m not sure how it is that you are doing it now, but if this is not automataised it is really easy that it gets outdated. Maybe worth looking into some read the docs automatised strategy.
Then if you like you can have additional information that has the core concepts, the troubleshooting, the FAQ and glossary.

This way the documentation might be a little bit more streamlined.
On the getting-started.md:
You have the requirements documentation in the main README.md, but then you also have it on your getting-started.md.The indications on each one are different so I would go for a single document and point to that solution in both cases.
On  the installation section, there is this message where people could get confused: “We recommend installing PyTorch first from pytorch.org” Maybe a quick screenshot of what is that they need to choose to get the pip install command.
On the installation documentation you would be missing the explanation for the conda creation.
Also, on the installation description or inside the code, it would be nice to let the user know if the GPU is correctly configured or if they will be using CPU (like on cellpose).
On step 3 (Quick start) do they need to put those code lines on a notebook or does the code go in a Python script? In any case, it would be better to have a quick start notebook or Python script than having people copy pasting code. Unless you want people to copy and paste these code lines into their already existing pipelines? Also the path to the images on that second step could be asked more interactively.
On the step 4 (Next Steps), the Interactive Quickstart should be the official quickstart for me instead of having the code cells on step 3. Also, maybe a link to Colab would be nicer (I know that while private this is not possible, but when you make it public).
On the step 4 (Next Steps), not sure if these links are next steps. They are relevant link, but glossary, core concepts or FAQ are not next steps.
On experiment-details.md
Links on the quick reference broken (Path Templates and Mask Discovery).
On 1 Path Templates, I’m not 100% sure what I should do. Can I put whatever I want? Also having a text input can be tricky as people could mess it up in so many ways, why not using input parameters: ordered list of with strings of drug/time/condition and extension of the file.
On 2 Using a CSV or Spreadsheet, you have the alternative approach that is related with the filename. For me that should be in the 1. Path Templates option. The rest of the information on the section is quite confusing and overwhelming.
Suddenly the masks section doesn't have numbers and there is an Object-Oriented Metadata that is not mentioned on the Quick Reference and gives some ideas but not details so not really clear for the users.
What I will do for sure in this section is to split it, it is too large. And many of the things are too advanced. I would have maybe something really simple of the simplest use case and then a link to a more advanced definition. Also, the troubleshooting section shouldn’t go there at the bottom, you should have a dedicated section for it.
On concepts.md:
Please do not use a code generated image on Framework Architecture. Is worth nothing to draw something on Google Draws or whatever you want.
Maybe it would be nice to have a figure for the processing flow.
Are ‘Temporal images’ and ‘Embedding Lifecycle’ really sections that fit on the ‘Framework architecture’? For me this should be for general concepts and this seems really specific.
Portability and Data Management is not mentioned on the table of concepts.
This documentation page I think that it starts being something with general concepts but at the end it starts becoming a little bit too technical with the code. I know that is not easy to find the balance and that at some point you would need to mention how it is used in the code. But I wouldn’t put it in the concepts one.
Also, maybe on things such as the dimensionality reduction documentation, you would benefit from having it in a separate documentation file that you can point to in your notebooks. For example, when choosing the dimensionality reduction method have something like: Not sure what method to use? See this extended documentation to understand the differences.
On faq.md:
On Data and Setup, you point to Getting Started: Expected Structure but there is nothing related to the expected structure on the getting started. And actually I think that it would be a nice step on the quick start, right after the installation.
On glossary.md:
If I were you, I would make each term linkable, this would allow you to point these explanations during the documentation or from the notebooks. Otherwise, you will have repeated explanations like on the correlation coefficient, that you have on the glossary and on the interpretability documentation: (https://github.com/AAitorG/PhenoMe/blob/main/docs/guides/interpretability.md#reading-correlation-coefficients). Otherwise, if you are not going to point this file, I would remove this glossary, it would be easier for you as you would not have duplicated explanations.
Rest of the files:
* I did not take a deep look into the rest of the document files because they are more oriented for advanced users and developers. Still, one thing that I feel is that there are some documents that are really long and it is not straightforward how to navigate through them. A quick example, on the “For Advanced Users & Developers” you point to Custom Properties and Extending the Framework. Then on the Extending the Framework I have again the Custom Properties section, so this is confusing.
About the length of the documents, a clear example is the custom property document, this is huge, maybe splitting it would make a easier navigation. Similar to the extending document, where you have a nice table pointing to the other resources and a quick explanation of what to expect on each one.
