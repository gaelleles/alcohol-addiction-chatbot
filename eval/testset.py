"""Gold evaluation set, grounded in pdf_papers/public/ so it's reproducible by
anyone who clones the repo (the private corpus isn't shipped).

Each case has:
- `reference`: the answer, restated from the source PDF. RAGAS' context_recall
  and context_precision use it as ground truth.
- `source` / `pages`: the PDF and the 1-indexed page(s) the answer comes from.
  Used for an LLM-free retrieval metric (was the right page retrieved?).
"""

WOMEN = (
    "alcohol-related-disparities-among-women-evidence-and-potential-explanations.pdf"
)
NATIVES = "Alcoholism_Natives_colonisation.pdf"
CAPITALISM = "048D16968153.pdf"
ATTITUDES = "attitudes_to_alcoholism.pdf"

TESTSET = [
    # --- Native Americans and alcohol ("perfect colonizer") ---
    {
        "question": (
            "How much higher is the rate of alcoholism among Native Americans "
            "compared to the U.S. average?"
        ),
        "reference": (
            "The rate of alcoholism among Native Americans is six times higher "
            "than the U.S. average, and one in every ten Native American deaths "
            "is a result of some alcohol-related cause."
        ),
        "source": NATIVES,
        "pages": [1],
    },
    {
        "question": (
            "Why is alcohol described by some Native American groups as the "
            "'perfect colonizer'?"
        ),
        "reference": (
            "Alcohol is called the 'perfect colonizer' because it has no "
            "conscience and shows no remorse for the modern-day devastation it "
            "has caused among Native American communities, unlike a human "
            "colonizer."
        ),
        "source": NATIVES,
        "pages": [1, 2],
    },
    {
        "question": (
            "What three-column model does the paper use to understand alcohol "
            "as the 'perfect colonizer'?"
        ),
        "reference": (
            "The paper models alcohol as the 'perfect colonizer' with a "
            "three-column model defined by the philosopher (Viola Cordova), the "
            "artist (Sherman Alexie) and the interventionist (Gene Thin Elk)."
        ),
        "source": NATIVES,
        "pages": [1, 2],
    },
    {
        "question": (
            "Why does the paper argue that the 12-Step Program works less well "
            "for Native Americans than the Red Road to Recovery?"
        ),
        "reference": (
            "The 12-Step Program emphasizes individualistic tenets (admission to "
            "self, submission to a higher being, personal inventory) that "
            "violate Native American beliefs, whereas the Red Road to Recovery "
            "emphasizes belonging to a caring circle of relatives, generosity "
            "toward the addict and responsibility to others, and welcomes the "
            "addict back into the group as soon as possible."
        ),
        "source": NATIVES,
        "pages": [5],
    },
    {
        "question": (
            "Where was alcohol consumption mostly confined among Native American "
            "groups before the 16th and 17th centuries?"
        ),
        "reference": (
            "Before the 16th and 17th centuries, alcohol and fermented beverages "
            "were primarily confined to the Native American groups of the "
            "southwestern United States, where the Mayans and Aztecs made "
            "fermented drinks such as balche and puique under ceremonial rules."
        ),
        "source": NATIVES,
        "pages": [1],
    },
    # --- Alcohol-related disparities among women ---
    {
        "question": (
            "Which groups of women are found to experience more alcohol-related "
            "disparities?"
        ),
        "reference": (
            "Racial/ethnic minority women, sexual minority women, and women of "
            "low socioeconomic status (based on education, income, or residence "
            "in disadvantaged neighborhoods) are more likely to experience "
            "alcohol-related problems."
        ),
        "source": WOMEN,
        "pages": [1],
    },
    {
        "question": (
            "Why does women's drinking warrant serious attention from "
            "researchers even though women consume less alcohol than men?"
        ),
        "reference": (
            "Women are more susceptible to certain alcohol-related problems at a "
            "given level of consumption, and women are less likely to receive "
            "help for problems with alcohol use."
        ),
        "source": WOMEN,
        "pages": [2],
    },
    {
        "question": (
            "According to the 2017 NSDUH data, how did alcohol dependence "
            "prevalence among bisexual and lesbian women compare with "
            "heterosexual women?"
        ),
        "reference": (
            "Among all women, 12-month alcohol dependence was 8.63% for bisexual "
            "women and 5.12% for lesbian women, versus 2.14% for heterosexual "
            "women."
        ),
        "source": WOMEN,
        "pages": [4],
    },
    {
        "question": "What is the 'alcohol harm paradox' described in the review?",
        "reference": (
            "Groups with greater socioeconomic advantages had similar or greater "
            "alcohol consumption than less advantaged groups, yet the less "
            "advantaged groups were at greater risk for alcohol-related "
            "problems. This is called the alcohol harm paradox."
        ),
        "source": WOMEN,
        "pages": [5],
    },
    {
        "question": (
            "How much higher is the risk of alcohol problems for women who "
            "drink and live in disadvantaged neighborhoods?"
        ),
        "reference": (
            "Women who drink and live in disadvantaged neighborhoods have "
            "twofold greater risk of alcohol problems (adjusted OR = 2.07 for "
            "two or more drinking consequences or DSM-IV alcohol dependence) "
            "than women who drink and live in more advantaged neighborhoods."
        ),
        "source": WOMEN,
        "pages": [5],
    },
    # --- Marxian and Durkheimian perspective on addiction ---
    {
        "question": (
            "According to the Marxian and Durkheimian perspective, how does "
            "market capitalism contribute to addiction?"
        ),
        "reference": (
            "Marxist concepts of self-estrangement and alienation frame market "
            "capitalism and labor distribution as negatively impacting "
            "individuals by supplying false needs and a sense of "
            "disillusionment; using labor as identity and emphasizing profit "
            "creates a disconnect between a person and their self, producing a "
            "culture of dissatisfaction."
        ),
        "source": CAPITALISM,
        "pages": [1, 2],
    },
    {
        "question": (
            "How does Durkheim's concept of anomie classify addiction, "
            "according to the article?"
        ),
        "reference": (
            "Durkheim's concept of anomie classifies addiction as a form of slow "
            "suicide caused by external social forces."
        ),
        "source": CAPITALISM,
        "pages": [1],
    },
    {
        "question": (
            "How do drugs and drug users disrupt the core ideologies of "
            "capitalism?"
        ),
        "reference": (
            "Drug users interrupt the flow of consumption: because addiction's "
            "only end game is the drug itself, all other products become "
            "useless. Drugs also provide the absolute enjoyment that ordinary "
            "merchandise only promises but never delivers."
        ),
        "source": CAPITALISM,
        "pages": [2],
    },
    {
        "question": (
            "What role does the deviant 'addict' identity play in capitalist "
            "society according to the functionalist theory used in the article?"
        ),
        "reference": (
            "The deviant addict identity plays a role in instigating and "
            "upholding certain ideologies of capitalist control and "
            "individualism."
        ),
        "source": CAPITALISM,
        "pages": [1],
    },
    # --- Attitudes towards alcohol dependence in Germany, 1990 vs 2011 ---
    {
        "question": (
            "When did the WHO define alcohol dependence as an illness, and why "
            "was this framing promoted?"
        ),
        "reference": (
            "The WHO defined alcohol dependence as an illness in the 1950s. "
            "Promoting this illness concept was meant to reduce the blame "
            "placed on individuals for their drinking problem and, "
            "consequently, reduce the stigma attached to the disorder."
        ),
        "source": ATTITUDES,
        "pages": [2],
    },
    {
        "question": (
            "What survey samples did the study on attitudes towards alcohol "
            "dependence in Germany use?"
        ),
        "reference": (
            "Two population surveys of German citizens aged 18 and over in the "
            "'old' German states, conducted in 1990 and 2011, using random "
            "subsamples of n = 1,022 (1990) and n = 1,167 (2011) for the "
            "alcohol dependence questions."
        ),
        "source": ATTITUDES,
        "pages": [1, 3],
    },
    {
        "question": (
            "What proportion of respondents agreed that alcohol dependence is "
            "an illness like any other, a weakness of character, or the "
            "individual's own fault?"
        ),
        "reference": (
            "About 55% agreed that alcohol dependence is an illness like any "
            "other, more than 40% said it was a weakness of character, and 30% "
            "said those affected are themselves to blame for their problems."
        ),
        "source": ATTITUDES,
        "pages": [1, 5],
    },
    {
        "question": (
            "Which attitude item changed the most between 1990 and 2011 in the "
            "German survey?"
        ),
        "reference": (
            "The biggest change concerned the item 'alcoholism is almost always "
            "resulting from psychological problems', which received less "
            "support (about -9%) and more opposition (about +3%) in 2011 than "
            "in 1990."
        ),
        "source": ATTITUDES,
        "pages": [5],
    },
    {
        "question": (
            "Which three factors emerged from the factor analysis of the seven "
            "attitude items?"
        ),
        "reference": (
            "Three factors emerged: 'blaming the individual', 'endorsing an "
            "illness concept' and 'trivializing the problem'."
        ),
        "source": ATTITUDES,
        "pages": [4, 5],
    },
    {
        "question": (
            "What did the review of anti-stigma initiatives find regarding "
            "substance use disorders?"
        ),
        "reference": (
            "A review of anti-stigma initiatives worldwide found that 59% were "
            "concerned with 'mental illness', 28% with schizophrenia and 9% "
            "with depression, and none with substance abuse disorders."
        ),
        "source": ATTITUDES,
        "pages": [5],
    },
]
