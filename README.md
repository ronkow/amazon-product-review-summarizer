# keyphrase-summarizer

As the Amazon product review dataset is too large, please download it from: 
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

Output for the product in the data instance above:
```
**** PRODUCT REVIEW SUMMARIZER ****

Index: 6806
Product ID: 120401325X
Number of reviews: 7

Sample reviews:

Review 1:
 They look good and stick good! I just don't like the rounded shape because I was always bumping it and Siri kept popping up and it was irritating. I just won't buy a product like this again

Review 2:
 These stickers work like the review says they do. They stick on great and they stay on the phone. They are super stylish and I can share them with my sister. :)

Review 3:
 These are awesome and make my phone look so stylish! I have only used one so far and have had it on for almost a year! CAN YOU BELIEVE THAT! ONE YEAR!! Great quality!


SUMMARY KEYPHRASES:

big deal
free screen protector
great deal
great quality
great time
look
multiple apple products
one year
phone look
stick good
super stylish
```
