# EQ2 : Can a detector trained on domain A perform well on texts from domain B (where domains A and B are topically dissimilar)?

So here we have data from 4 domains:
Covid-19
SBIRs
News articles
webtext

We finetune BERT 4 times separately on each domain.
We test performance of each on all the domains.

Runners:

<ol>

    <li> Finetune on Covid-19 and test on:
        <ul>
             <li> SBIRs</li>
            <li> News articles</li>
            <li> webtext</li>
        </ul>
    </li>

    <li> Finetune on SBIRs and test on:
        <ul>
            <li> Covid-19</li>
            <li> News articles</li>
            <li> webtext</li>
        </ul>
    </li>

    <li> Finetune on News articles and test on:
        <ul>
            <li> Covid-19</li>
            <li> SBIRs</li>
            <li> webtext</li>
        </ul>
    </li>

    <li> Finetune on Webtext and test on:
        <ul>
            <li> Covid-19</li>
            <li> SBIRs</li>
            <li> News articles</li>
        </ul>
    </li>

</ol>