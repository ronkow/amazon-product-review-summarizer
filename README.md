# Amazon product review summarizer

This application extracts information from Amazon product reviews to summarize the content and sentiments of the reviews.

Please read the project report for details:  
https://github.com/ronkow/amazon-review-summarizer/blob/main/project_report/project_report.pdf

As the Amazon product review dataset is too large, please download it from: 
#### http://jmcauley.ucsd.edu/data/amazon/index_2014.html

We use the following dataset from the section **"Small" subsets for experimentation**:  
**Cell Phones and Accessories 5-core** (194,439 reviews)


A data instance:
```
{
    "reviewerID": "A3U8HK7BNWPB53", 
    "asin": "B00AA6CS86", 
    "reviewerName": "Alex Limon", 
    "helpful": [0, 1], 
    "reviewText": "Very portable. It works perfectly for my needs (quick trips). Now I have an 
                  additional full charge when is not a power source near by.", 
    "overall": 5.0, 
    "summary": "great", 
    "unixReviewTime": 1398297600, 
    "reviewTime": "04 24, 2014"
}
```

Output for the product in the data instance above:
```
**** PRODUCT REVIEW SUMMARIZER ****

Index: 1005
Product ID: B00AA6CS86
Number of reviews: 38

Sample reviews:

Review 1:
 Very portable. It works perfectly for my needs (quick trips). Now I have an 
 additional full charge when is not a power source near by.

Review 2:
 The size is great, not too heavy. Purchased to use on vacation to use my iphone 
 as my camera and be able to recharge while out and about. It worked perfectly a 
 couple of times. The third time I plugged in my phone or ipad, and no response, 
 nothing. This was after it had been fully charged the day before, then put in my 
 bag, unused. It's supposed to hold a charge for 6 months. I plugged the jackery 
 into ac to charge and it was at 3 of 4. It then worked fine for the rest of the 
 day. Had same issue a couple of days later. Not too handy if it inexplicably won't 
 work occasionally. I don't know if I got a bad one.Currently deciding whether to 
 return it or contact jackery for a replacement.When it did work it was fantastic 
 - quickly and fully recharged my iphone 5 with a little bit of charge to spare. 
 - Definitely can't get much more than one full charge out of it. I also used it 
 - a couple of times to  give a small boost charge to my ipad.

Review 3:
 Read my review under the Jackery Giant 10400mAh.  Love this charger.  I've purchased 
 3 Jackery products and would recommend all three.


SUMMARY KEYPHRASES:

battery power level
compact charger fits
enough charge capacity
first mini jackery
full charge
great emergency charger
iphone charging cable
jackery mini
jackery mini gold
phone
phone battery get
portable battery charger
portable charger
power pack fit
small battery pack
very nice charger
```
