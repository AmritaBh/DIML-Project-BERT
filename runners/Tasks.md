# EQ2 : Can a detector trained on domain A perform well on texts from domain B (where domains A and B are topically dissimilar)?

So here we have data from 4 domains:
Covid-19
SBIRs
News articles
webtext

We finetune BERT 4 times separately on each domain.
We test performance of each on all the domains.

Runners:

1. Finetune on Covid-19 and test on:
    a. SBIRs
    b. News articles
    c. Webtext

2. Finetune on SBIRs and test on:
    a. Covid-19
    b. News articles
    c. Webtext

3. Finetune on News articles and test on:
    a. Covid-19
    b. SBIRs
    c. Webtext

4. Finetune on Webtext and test on:
    a. Covid-19
    b. SBIRs
    c. News articles
