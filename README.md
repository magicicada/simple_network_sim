# simple_network_sim
Adaptation of a simple network simulation model to COVID-19 (forked from https://github.com/magicicada/simple_network_sim to be brought into the SCRC consortium GitHub organisation - this is the main repository for development). Similar models have previously been used to model other disease outbreaks with different characteristics to COVID-19.

## Features
*TODO: What does the software do? How does it compare with other similar software? May want to refer to a seperate concept of operations document.*

## Contributing
> **Rule 0:** Be polite, respectful and supportive.

At present, contributions are limited to members of Scottish COVID-19 Response Consortium (SCRC). The process for implementing changes is:
- An **issue** is raised in the [SCRC issue tracking repository](https://github.com/ScottishCovidResponse/SCRCIssueTracking), tagged "Simple Network Sim"
> *How to write a good issue:* Check for duplicate issues - don't raise the same issue twice! Be specific - whoever works on this needs to know where it ends. Not too big - if the issue could be multiple issues, it probably should be.
- The **issue** is assigned (ideally self assigned with the agreement of the team) to a team member (assignee)
- The assignee creates a **branch** to work on the issue
- The assignee makes a series of **commits** to the branch, implementing changes that address the issue
- The assignee makes a **pull request**, requesting a **review** (a review is required to merge, and a request lets the team member know to do a review)
> *How to do a good pull request:* Tie it to usually one or sometimes more issues - e.g. write 'resolves ScottishCovidResponse/SCRCIssueTracking#1'. This will automatically mark the issue as closed when the pull request is merged. Summarise what you've done to resolve the issue(s).

**For the time being, all pull requests must be reviewed by @magicicada or @bobturneruk.**
- The assignee (or reviewer) merges the pull request into the `master` branch

## Installation
This software requires Python (*version?*) which can be installed using (Anaconda)[https://www.anaconda.com/products/individual] or another distribution. It also needs the `networkx` and `matplotlib` packages which can be installed via Anaconda or using (pypi)[https://pypi.org/].

## Usage
To run a example case, enter the following at the command prompt:

```{shell}
python sampleUseOfModel.py sample_input_files/paramsAgeStructured sample_input_files/sample_hb2019_pop_est_2018.sampleCSV sample_input_files/sample_scotHB_commute_moves_wu01.sampleCSV afilename.pdf
```

## License
The 2-Clause BSD License