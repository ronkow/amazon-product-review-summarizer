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

Sample output:
```
**** PRODUCT REVIEW SUMMARIZER ****

Index: 6807
Product ID: B009EARUFU
Number of reviews: 7

Sample reviews:

Review 1:
 Came on time and as described. The color isn't really the color in the picture though. I don't really like it but I guess that's my problem now! I've seen the teal one though in person, if I were you I'd get that one! Much prettier!

Review 2:
 The color is lighter than what is pictured, but it's still pretty.  Case is very snug and seems like it will protect my iPhone 5 well.  The only complaint I have is that the case makes it hard to press down on the power button, but other than that it is a great case for the price.*UPDATE*: After having the case for a couple of weeks I have found that it gets very dirty! and any dirt, stains, or color transfer you get on it cannot be cleaned off.  So disappointing! I'd recommend getting the black one. I will probably be ordering that one soon.

Review 3:
 great quality, and its so cool that it is textured. fits my iphone 5 perfectly. its a really awesome case


SUMMARY KEYPHRASES:

awesome case
case
fit silicon case
great case
iphone 5 case
not write reviews
one though
phone
phone case
really awesome case
totally worth buying
very nice iphone
```
