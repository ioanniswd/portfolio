### The problem
Athens, Greece is densely populated. It has become a real challenge to find an
apartment in general, and even more so to find one in a quiet neighborhood.
<br />
<br />
I tried to think of a way to search for less crowded areas in Athens. This poses
a challenge, as public data on population density are severely affected by the
residence each person declares as their main one.
For example, an owner of an apartment in Athens might also have a residence in a
remote area, which they register as their main one, while either themselves or
possibly a relative may be staying in the aforementioned apartment in Athens all
year round.
<br />
<br />
### The approach
To overcome this challenge, I decided to use the number of schools in each area
as a proxy for population density. The idea is that the more schools there are
in an area, the more families with children live there. This is a good indicator
of population density, albeit not perfect.
<br />
<br />

### The results
Here is the map of Athens with the schools marked on it:
![scatter_map_schools_athens](./scatter_map_schools_athens_gov.png "Scatter Plot")

Disregarding some areas like Piraeus where the data seem to be lacking, and
other which are mainly industrial, we can see that population density is high
throughout the city, with some exceptions. As it would be expected, the areas with
less dense population are the more "prestigious" ones, like Ekali, Glyfada and
Voula.
<br />
<br />
### The conclusion
When peace and quiet are a priority in Athens, it is best to look for an
apartment or a house at the outskirts of the city, preferably in the south,
which is closer to the cluster of less densely populated and prestigious areas
of Glyfada, Voula and the under development area of Elliniko, so as to still be
within driving distance from the city center and also be within a reasonable
budget. The proximity of the Stavros Niarchos Foundation Cultural Center is also
a big plus for the southern suburbs.

### The details
For the first iteration of this project, I scraped a list of schools in Athens
from a .gov website. I then used the Google Maps API to get the coordinates of
each school based on its address.
<br />
For the next iteration, I used an official dataset of schools in Greece, which
I found on the Greek government's open data portal.
