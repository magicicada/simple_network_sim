# Example plans for simulating scenarios using simple network sim

The main mechanism for simulating scenarios of future infection using simple network sim is via modificaiton of one or more of the inputs to the model.  We give a number of examples of such approaches here.  These are not meant to provide the best possible realism or to be exhaustive, but instead to give general exposition on how and why the inputs to simple network sim might be modified. 

##### Scenario: increased school activity

Basic approach: Provided that schools operate mainly within the geographic units that are captured by nodes, the simplest approach would be to increase the mixing of school-age people in the mixing matrix input.  Limitations include the approximation of all schools within a region as a homogeneously mixing population.  A more complex approach if capturing schools more accurately were considered essentialy might be to produce a new network in which the nodes are school catchment areas.  

##### Scenario: increased overall mixing or between-region movements

Basic approach: Simulating a uniform general change in the volume of movements or contacts within regions can be done by changing the time-varying modifiers captured in the movement modifier and contact modifier inputs

##### Scenario: modelling testing and isolation

Basic approach: There are several simple ways of approximately modelling a test-and-isolate system.  First, one might change the contact modifier: since the outcome of the model is epidemiological, the contact modifiers only impact the outcome when they are applied to contact from an infected person.  Secondly (and perhaps more informatively), one could implement a modified compartmental model in which there is a comparment or compartments of tested-and-isolated individuals, who can progress through the disease to recovered, hospitalised, dead, in the usual way, but are not considered infectious.  

##### Scenario: modelling care homes more explicitly

Basic approach: Again, there are several straightforward options for approximately modelling care homes within simple network sim.  One might add care homes as their own node(s) with between-node contact accounting for visitors, staff, etc.  Less explicitly and perhaps less conveniently, one could add a column and row to the mixing matrix (as well as the population table and the compartmental transitions) that is not an age category, but is instead a particular group of people - in this case care home residents.  Then mixing between the residents and the outside world as well as within the group of residents in a node's region would be managed by the mixing matrix.