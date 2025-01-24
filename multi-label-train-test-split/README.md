My girlfriend's master's thesis was the `Perception of Anthropomorphic Traits in
Cars`. She wanted to create a questionaire to test this hypothesis.
<br />
<br />
She had acquired a dataset of images of cars, and after selecting the images
that could be used in the questionaire, she created a spreadsheet with the file name
and the features for each car, such as the size of the grille, the shape of the
headlights, etc.
<br />
<br />
She needed to select 10 images for the questionaire, and those images had to
be representative of the different classes of the various labels, e.g. `Bumper
Shape: upturned lower edge-straight upper edge` or `Headlights Position: only
upper`.
<br />
<br />
To tackle this, I used [scikit-multilearn](http://scikit.ml/) to split the stimuli into train and test sets where all labels were represented, and used one of them for the questionaire.
