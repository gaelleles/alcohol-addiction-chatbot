"""Small gold evaluation set, grounded in pdf_papers/public/ so it's reproducible
by anyone who clones the repo (the private corpus isn't shipped).

Each `reference` is a fact taken directly from the source PDF, used by RAGAS'
context_recall and context_precision metrics as the ground truth to check
retrieval against.
"""

TESTSET = [
    {
        "question": "How much higher is the rate of alcoholism among Native Americans compared to the U.S. average?",
        "reference": (
            "The rate of alcoholism among Native Americans is six times higher than the U.S. average, "
            "and one in every ten Native American deaths is a result of some alcohol-related cause."
        ),
    },
    {
        "question": "Why is alcohol described by some Native American groups as the 'perfect colonizer'?",
        "reference": (
            "Alcohol is called the 'perfect colonizer' because it has no conscience and shows no remorse "
            "for the modern-day devastation it has caused among Native American communities, unlike a "
            "human colonizer."
        ),
    },
    {
        "question": "Which groups of women are found to experience more alcohol-related disparities?",
        "reference": (
            "Racial/ethnic minority women, sexual minority women, and women of low socioeconomic status "
            "(based on education, income, or residence in disadvantaged neighborhoods) are more likely to "
            "experience alcohol-related problems."
        ),
    },
    {
        "question": "Why does women's drinking warrant serious attention from researchers even though women consume less alcohol than men?",
        "reference": (
            "Women are more susceptible to certain alcohol-related problems at a given level of consumption, "
            "and women are less likely to receive help for problems with alcohol use."
        ),
    },
    {
        "question": "When did the WHO define alcohol dependence as an illness, and why was this framing promoted?",
        "reference": (
            "The WHO defined alcohol dependence as an illness in the 1950s. Promoting this illness concept "
            "was meant to reduce the blame placed on individuals for their drinking problem and, "
            "consequently, reduce the stigma attached to the disorder."
        ),
    },
    {
        "question": "According to the Marxian and Durkheimian perspective, how does market capitalism contribute to addiction?",
        "reference": (
            "Marxist concepts of self-estrangement and alienation frame market capitalism and labor "
            "distribution as negatively impacting individuals by supplying false needs and a sense of "
            "disillusionment; using labor as identity and emphasizing profit creates a disconnect between "
            "a person and their self, producing a culture of dissatisfaction."
        ),
    },
]
