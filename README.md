# keyphrase-summarizer

Download the Amazon data from: 
#### http://jmcauley.ucsd.edu/data/amazon/index_2014.html

We use the following dataset from the section **"Small" subsets for experimentation**:  
**Cell Phones and Accessories 5-core** (194,439 reviews)


A data instance:
```
{    
    "reviewerID": "A30TL5EWN6DFXT", 
    "asin": "120401325X", 
    "reviewerName": "christina", 
    "helpful": [0, 0], 
    "reviewText": "They look good and stick good! I just don't like the rounded shape because I was always bumping it and Siri kept popping up and it was   
        irritating. I just won't buy a product like this again", 
    "overall": 4.0, 
    "summary": "Looks Good", 
    "unixReviewTime": 1400630400, 
    "reviewTime": "05 21, 2014"
}
```

Sample output:
```
**** PRODUCT REVIEW SUMMARIZER ****

Index: 1001
Product ID: B0087GALZI
Number of reviews: 38

Sample reviews:

Review 1:
 Everyone asks about it, it's just right for my GS3 and protects it well. Would buy again , love the style and color

Review 2:
 Fits perfect on the cell phone and I love the design and colors.  Doesn't slip so stays put.  Fast and packaged good.

Review 3:
 I am impressed with this cover/case, I purchased two, one purple and one clear. They are both consistent in the way they fit my phone. I also purchased the Incipio SA-301 Feather Ultra-Light Hard Shell Case, which I love, but I feel a bit more secure with the Cimo in regards to dropping the phone and it landing face down. The Cimo has more of a 'rim' of protection around the front and it seems it may provide enough height that it could save the face of the phone. The cover/case has a nice feel, not a gummy or dull appearance, and it is flexible while still giving 'hard' case protection.


SUMMARY KEYPHRASES:

case
case fit
cruzerlite tpu case
different tpu case
matte tpu case
much better case
nice simple case
phone
silicon feeling case
slim fit case
smoke color case
softest tpu case
tpu case
tpu case makers
transparent cover case
very good case
white phone
```
