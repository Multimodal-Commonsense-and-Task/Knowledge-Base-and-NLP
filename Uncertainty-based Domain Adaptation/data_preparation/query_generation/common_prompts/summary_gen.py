from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables.retry import RunnableRetry
from difflib import SequenceMatcher
import tiktoken

# from langchain_core.output_parsers import JsonOutputParser

# json_parser = JsonOutputParser()

# output_force_prompt_v3 = """
# Output ONLY valid JSON without any additional text or explanation. Do not include any text before or after JSON. Output Example: {"queries": [query1, query2, ... query20]}
# Output: """


def remove_intro_line(text):
    lines = text.strip().split("\n", 1)
    # Check if the first line contains "Here is"
    if "here is" in lines[0].lower():
        # Remove the first line
        text = lines[1] if len(lines) > 1 else ""
    return text.strip()


def bool_decide(text):
    last_line = text.split("\n")[-1]
    if "yes" in last_line.lower().strip():
        return True
    elif "no" in last_line.lower().strip():
        return False
    else:
        first_chars = text[:5]
        if "yes" in first_chars.lower().strip():
            return True
        elif "no" in first_chars.lower().strip():
            return False
    raise ValueError(f"Invalid response: {text}")


class SummaryDecisionOutputParser(BaseOutputParser[bool]):
    def parse(self, text):
        try:
            return bool_decide(text)
        except ValueError as e:
            raise OutputParserException(str(e))

    @property
    def _type(self):
        return "summary_decision_output_parser"


FEWSHOT_1H_DOC = """Example 1:
Document: You\'re missing a very important thing: YEAR END values in (U.S.) $ millions unless otherwise noted So 7098 is not $7,098.  That would be a rather silly amount for Coca Cola to earn in a year don\'t you think?  I mean, some companies might happen upon random small income amounts, but it seems pretty reasonable to assume they\'ll earn (or lose) millions or billions, not thousands. This is a normal thing to do on reports like this; it\'s wasteful to calculate to so many significant digits, so they divide everything by 1000 or 1000000 and report at that level.  You need to look on the report (usually up top left, but it can vary) to see what factor they\'re dividing by. Coca Cola\'s earnings per share are $1.60 for FY 2014, which is 7,098/4450 (use the whole year numbers, not the quarter 4 numbers; and here they\'re both in millions, so they divide out evenly).   You also need to understand that ""Dividend on preferred stock"" is not the regular dividend; I don\'t see it explicitly called out on the page you reference. They may not have preferred stock and/or may not pay dividends on it in excess of common stock (or at all)."""
FEWSHOT_1H_SENT = f"""{FEWSHOT_1H_DOC}
Sentence: Coca Cola\'s earnings per share are $1.60 for FY 2014, which is 7,098/4450 (use the whole year numbers, not the quarter 4 numbers; and here they\'re both in millions, so they divide out evenly).
Relevant Query: """
FEWSHOT_1H = f"""{FEWSHOT_1H_DOC}
Relevant Query: """
FEWSHOT_1H_KW = f"""{FEWSHOT_1H_DOC}
Keywords: 1.Year end values 2.Earnings per share 3.Dividend on preferred stock
Relevant Query: """
FEWSHOT_1H_KWGEN = f"""{FEWSHOT_1H_DOC}
Keywords: """
FEWSHOT_1H_SM = f"""{FEWSHOT_1H_DOC}
Summary: The document emphasizes the importance of understanding that financial reports often present figures in units like millions or billions. It explains that Coca-Cola's reported income of "7098" actually means $7,098 million, not $7,098. The author highlights the need to adjust calculations accordingly, especially when computing earnings per share, using whole-year numbers in millions. Additionally, the document clarifies that "Dividend on preferred stock" is different from regular dividends and may not apply if the company doesn't have preferred stock or doesn't pay dividends on it.
Relevant Query: """
FEWSHOT_1H_SUMMARY = """Example 1:
Summary: The document emphasizes the importance of understanding that financial figures in reports, like Coca-Cola's, are often presented in millions or billions of dollars rather than smaller units, to avoid unnecessary detail. It suggests checking the report for any indication of unit scaling (usually in the top left corner). For instance, Coca-Cola’s earnings per share for FY 2014 are calculated as $1.60, derived from $7,098 million earnings divided by 4,450 million shares. Additionally, it clarifies that "Dividend on preferred stock" is different from regular dividends, and Coca-Cola may not pay preferred stock dividends if they lack preferred shares or choose not to distribute extra dividends on them.
Relevant Query: """
FEWSHOT_1H_STREAMLINED = """Example 1:
Clarified Document: When reviewing financial reports, remember that figures are often shown in millions or billions of dollars to simplify reporting. For example, Coca-Cola’s earnings per share for FY 2014 are $1.60, calculated by dividing $7,098 million in earnings by 4,450 million shares. Be sure to check the report (usually in the top left) to see the unit scale. Also, "Dividend on preferred stock" is not the regular dividend, and some companies may not pay it if they lack preferred stock or choose not to issue additional dividends on it.
Relevant Query: """
FEWSHOT_1A = "How do I find out the Earnings Per Share of a Coca Cola Co Share?"
FEWSHOT_1A_SUMMARY = "How is Coca-Cola's earnings per share calculated from scaled financial figures in its reports?"
FEWSHOT_1A_STREAMLINED = "What is the importance of checking the unit scale in financial reports when calculating earnings per share?"
FEWSHOT_1A_KWGEN = (
    "1.Year end values 2.Earnings per share 3.Dividend on preferred stock"
)

# FEWSHOT_1H_TRECCOVID = """Example 1:
# Document: Ileostomy for steroid-resistant acute graft-versus-host disease of the gastrointestinal tract Steroid-resistant acute graft-versus-host disease (GVHD) of the gastrointestinal tract associates with important morbidity and mortality. While high-dose steroids are the established first-line therapy in GVHD, no second-line therapy is generally accepted. In this analysis of 65 consecutive patients with severe, steroid-resistant, intestinal GVHD (92% stage 4), additional ileostomy surgery significantly reduced overall mortality (hazard ratio 0.54; 95% confidence interval, 0.36\u20130.81; p = 0.003) compared to conventional GVHD therapy. Median overall survival was 16 months in the ileostomy cohort compared to 4 months in the conventional therapy cohort. In the ileostomy cohort, both infectious- and GVHD-associated mortality were reduced (40% versus 77%). Significantly declined fecal volumes (p = 0.001) after surgery provide evidence of intestinal adaptation following ileostomy. Correlative studies indicated ileostomy-induced immune-modulation with a > 50% decrease of activated T cells (p = 0.04) and an increase in regulatory T cells. The observed alterations of the patients\u2019 gut microbiota may also contribute to ileostomy\u2019s therapeutic effect. These data show that ileostomy induced significant clinical responses in patients with steroid-resistant GVHD along with a reduction of pro-inflammatory immune cells and changes of the intestinal microbiota. Ileostomy is a treatment option for steroid-resistant acute GVHD of the gastrointestinal tract that needs further validation in a prospective clinical trial. ELECTRONIC SUPPLEMENTARY MATERIAL: The online version of this article (10.1007/s00277-019-03754-3) contains supplementary material, which is available to authorized users.
# Scientific Query: """
# FEWSHOT_1A_TRECCOVID = "Does ileostomy improve survival outcomes and modulate immune responses in patients with steroid-resistant acute gastrointestinal graft-versus-host disease (GVHD)?"

# FEWSHOT_1H_TOUCHE = """Example 1:
# Document: Illegal Immigration The U.S.A. is thought to be the land of opportunity, the land of freedom. Other countries should follow their example. The world should provide equal rights and opportunities to everyone. I believe that the whole world should allow immigrants a chance to apply for citizenship. Sometimes, there is simply no other way for immigrants. This is their only chance to get a better life. Some people like us won the \"human lottery\". We have everything we need to survive and prosper. Some other people simply lost. They have no way to live well in their current condition. For example, in Mexico, it is very hard to get a job without a college degree because there are so many of those people, and not many jobs. Also, the living conditions are terrible. These type of people have almost no chance to live well in Mexico. You might hear all the stereotypes about very violent people that deserve to be deported, but that is not true. For example,only 1.6% of illegal immigrants in the U.S. commit crimes, which is actually less than the overall U.S. crime rate, which is about 2.8%. Most immigrants are hard-working people who just want a better life for their family. Even if immigrants do commit crimes, then they would be put in jail/prison and then deported. That is how the process works for legal immigrants already. More immigrants will benefit the country, not hurt it. Also, if immigrants have kids in the country, then those kids will have to be split up from their families. No kids should have to undergo that hurt. Illegal immigrants usually are good people trying to get a better life. We should let them.
# Relevant Query: """
# FEWSHOT_1A_TOUCHE = "What are the arguments for allowing illegal immigrants to apply for citizenship, including impacts on crime rates, family separation, and economic benefits?"

FEWSHOT_1H_SCIFACT = """Example 1:
Document: Can genome engineering be used to target cancer-associated enhancers? Transcriptional misregulation is involved in the development of many diseases, especially neoplastic transformation. Distal regulatory elements, such as enhancers, play a major role in specifying cell-specific transcription patterns in both normal and diseased tissues, suggesting that enhancers may be prime targets for therapeutic intervention. By focusing on modulating gene regulation mediated by cell type-specific enhancers, there is hope that normal epigenetic patterning in an affected tissue could be restored with fewer side effects than observed with treatments employing relatively nonspecific inhibitors such as epigenetic drugs. New methods employing genomic nucleases and site-specific epigenetic regulators targeted to specific genomic regions, using either artificial DNA-binding proteins or RNA-DNA interactions, may allow precise genome engineering at enhancers. However, this field is still in its infancy and further refinements that increase specificity and efficiency are clearly required.
Claim Supported By Document: """
FEWSHOT_1H_SENT_SCIFACT = """Example 1:
Document: Can genome engineering be used to target cancer-associated enhancers? Transcriptional misregulation is involved in the development of many diseases, especially neoplastic transformation. Distal regulatory elements, such as enhancers, play a major role in specifying cell-specific transcription patterns in both normal and diseased tissues, suggesting that enhancers may be prime targets for therapeutic intervention. By focusing on modulating gene regulation mediated by cell type-specific enhancers, there is hope that normal epigenetic patterning in an affected tissue could be restored with fewer side effects than observed with treatments employing relatively nonspecific inhibitors such as epigenetic drugs. New methods employing genomic nucleases and site-specific epigenetic regulators targeted to specific genomic regions, using either artificial DNA-binding proteins or RNA-DNA interactions, may allow precise genome engineering at enhancers. However, this field is still in its infancy and further refinements that increase specificity and efficiency are clearly required.
Sentence: Transcriptional misregulation is involved in the development of many diseases, especially neoplastic transformation. Distal regulatory elements, such as enhancers, play a major role in specifying cell-specific transcription patterns in both normal and diseased tissues, suggesting that enhancers may be prime targets for therapeutic intervention.
Claim Supported By Document: Can genome engineering be used to target cancer-associated enhancers?"""
FEWSHOT_1A_SCIFACT = "Genome engineering holds potential for targeting cancer-associated enhancers, offering a pathway to restore normal gene regulation with fewer side effects than nonspecific epigenetic drugs."

# FEWSHOT_1H_NFCORPUS = """Example 1:
# Document: A modest proposal for a longitudinal study of dementia prevention (with apologies to Jonathan Swift, 1729). Many studies have documented the role of risk and protective factors for late life dementing illnesses, particularly Alzheimer's disease. A \"Systematic Review\" from the US Agency for Healthcare Research and Quality and the National Institute on Aging concluded that because the overall quality of evidence was low, recommendations for public health could not be made. In order to gain evidence for the efficacy of lifestyle interventions, we propose a \"Modest Proposal\" to study 10,000 subjects over 40 years randomly assigned to groups of low or high saturated fat in the diet, head injury, and high or low levels of mental activity, physical activity, or inactivity as well as smoking or non-smoking. This proposed study cannot be accomplished. The \"Modest Proposal\" illustrates that the absence of definitive evidence should not restrict physicians from making reasonable recommendations based on the evidence that is available.
# Scientific Query: """
# FEWSHOT_1A_NFCORPUS = "What is the current evidence on the effectiveness of lifestyle interventions, such as diet, physical activity, and smoking cessation, in preventing dementia and Alzheimer’s disease in older adults?"

FEWSHOT_1H_ARGUANA = """Example 1:
Document: ployment tax education university house would fund provision higher education The main problem with the proposition argument is the belief that a graduate will be earning \u00a340,000 immediately after leaving university, this is clearly not the case, particularly in the current economic climate, the average starting wage for a graduate was in 2009 \u00a323,500 with only one in ten exceeding \u00a336,000. (Milkround, 2009) The argument does in part accept this weakness however what it does not point out is that many careers which require a university degree may never pay greater than \u00a340,000. What a graduate tax focuses on is getting a job after university, this is not always the reason that people wish to go to university, take for example a mature student who just wants to self-better themselves, could they still get access to education when the system would be built upon getting young people into work? University should not be commoditized, it should be considered sacred in its own right; introducing a graduate tax turns university into a means to get a career rather than being a place of pure education.
Counter Argument:
"""
FEWSHOT_1H_SENT_ARGUANA = """Example 1:
Document: ployment tax education university house would fund provision higher education The main problem with the proposition argument is the belief that a graduate will be earning \u00a340,000 immediately after leaving university, this is clearly not the case, particularly in the current economic climate, the average starting wage for a graduate was in 2009 \u00a323,500 with only one in ten exceeding \u00a336,000. (Milkround, 2009) The argument does in part accept this weakness however what it does not point out is that many careers which require a university degree may never pay greater than \u00a340,000. What a graduate tax focuses on is getting a job after university, this is not always the reason that people wish to go to university, take for example a mature student who just wants to self-better themselves, could they still get access to education when the system would be built upon getting young people into work? University should not be commoditized, it should be considered sacred in its own right; introducing a graduate tax turns university into a means to get a career rather than being a place of pure education.
Sentence: What a graduate tax focuses on is getting a job after university, this is not always the reason that people wish to go to university, take for example a mature student who just wants to self-better themselves, could they still get access to education when the system would be built upon getting young people into work?
Counter Argument:
"""
FEWSHOT_1A_ARGUANA = "Since university graduates generally earn higher incomes, they should pay a graduate tax to fund higher education, ensuring that those who benefit most from their degrees contribute to making university accessible for future students. This system would also encourage students to pursue degrees that lead to well-paying jobs, aligning higher education with workforce demands and helping create a sustainable funding model."

FEWSHOT_1H_ROBUST04 = """Example 1:
Document: December 25, 1990, Tuesday, Orange County Edition\nA mobile-home fire that killed an elderly woman Sunday night was accidental and\nstarted in her bed, Orange County Fire Department officials said Monday.\n\"Some sort of smoking materials in the bedding ignited the fire,\" said Kathleen\nCha, a County Fire Department spokeswoman.\nThe 75-year-old woman, whose name has been withheld pending notification of\nrelatives, died of smoke inhalation, according to a report from Deputy Coroner\nBill King. She was a cigarette smoker and had been suffering from an\nundetermined illness, said Leah Lindsay, a neighbor at Dana Point Marina Mobile\nHome Estates.\nThe fire, which started about 6:30 p.m. Sunday, began in a rear bedroom and\nquickly spread throughout the mobile home, Cha said.\nThe victim apparently became disoriented during the fire and headed toward her\ndressing room and bathroom area, where she was found on the floor, Cha said.\n\"She was probably disoriented because there was no exit that way,\" Cha said.\n\"We recommend that mobile-home owners install a back exit -- either make one\nfor themselves or get one built in.\"\nFirefighters quickly brought the blaze under control and prevented it from\nspreading to other mobile homes in the 100-home park near Dana Point Harbor.\nA dead cat, which fire officials assume belonged to the victim, was also found\nin charred remains of the home, Cha said. She added that no smoke detector was\nfound in the charred home.\n\"All mobile homes should have smoke detectors, particularly because of the\nmaterials the homes are made of,\" Cha said. \"They are very combustible. We\nestimate it takes only three to five minutes for a mobile home to become fully\nengulfed.\"\nLindsay said the victim, a friend of 45 years, had lived in the park for 13\nyears after moving from Toluca Lake.\n\"Most of her friends had moved away,\" Lindsay said. \"We know she had some\nbrothers and nephews in Las Vegas, but we have not been able to locate them.\"\nThe victim's recent illness had kept her at home, Lindsay said.\n\"She's been ill for the past three months -- some sort of internal problem that\nthe doctors have not been able to figure out,\" Lindsay said. \"That kept her\nhome and in bed quite a lot.\"\nLindsay said she last saw the woman on the day of the fire.\n\"I took her to the store and then brought her some soup later,\" Lindsay said.\n\"It's really tragic that it happened, especially at this time of the year.\" LEN\nHALL", "title": "ORANGE COUNTY FOCUS: DANA POINT;\nFATAL MOBILE-HOME FIRE STARTED IN BED
Relevant Query: """
FEWSHOT_1H_SENT_ROBUST04 = """Example 1:
Document: December 25, 1990, Tuesday, Orange County Edition\nA mobile-home fire that killed an elderly woman Sunday night was accidental and\nstarted in her bed, Orange County Fire Department officials said Monday.\n\"Some sort of smoking materials in the bedding ignited the fire,\" said Kathleen\nCha, a County Fire Department spokeswoman.\nThe 75-year-old woman, whose name has been withheld pending notification of\nrelatives, died of smoke inhalation, according to a report from Deputy Coroner\nBill King. She was a cigarette smoker and had been suffering from an\nundetermined illness, said Leah Lindsay, a neighbor at Dana Point Marina Mobile\nHome Estates.\nThe fire, which started about 6:30 p.m. Sunday, began in a rear bedroom and\nquickly spread throughout the mobile home, Cha said.\nThe victim apparently became disoriented during the fire and headed toward her\ndressing room and bathroom area, where she was found on the floor, Cha said.\n\"She was probably disoriented because there was no exit that way,\" Cha said.\n\"We recommend that mobile-home owners install a back exit -- either make one\nfor themselves or get one built in.\"\nFirefighters quickly brought the blaze under control and prevented it from\nspreading to other mobile homes in the 100-home park near Dana Point Harbor.\nA dead cat, which fire officials assume belonged to the victim, was also found\nin charred remains of the home, Cha said. She added that no smoke detector was\nfound in the charred home.\n\"All mobile homes should have smoke detectors, particularly because of the\nmaterials the homes are made of,\" Cha said. \"They are very combustible. We\nestimate it takes only three to five minutes for a mobile home to become fully\nengulfed.\"\nLindsay said the victim, a friend of 45 years, had lived in the park for 13\nyears after moving from Toluca Lake.\n\"Most of her friends had moved away,\" Lindsay said. \"We know she had some\nbrothers and nephews in Las Vegas, but we have not been able to locate them.\"\nThe victim's recent illness had kept her at home, Lindsay said.\n\"She's been ill for the past three months -- some sort of internal problem that\nthe doctors have not been able to figure out,\" Lindsay said. \"That kept her\nhome and in bed quite a lot.\"\nLindsay said she last saw the woman on the day of the fire.\n\"I took her to the store and then brought her some soup later,\" Lindsay said.\n\"It's really tragic that it happened, especially at this time of the year.\" LEN\nHALL", "title": "ORANGE COUNTY FOCUS: DANA POINT;\nFATAL MOBILE-HOME FIRE STARTED IN BED
Sentence: A mobile-home fire that killed an elderly woman Sunday night was accidental and\nstarted in her bed, Orange County Fire Department officials said Monday.
Relevant Query: """
FEWSHOT_1A_ROBUST04 = (
    "What caused the fatal mobile-home fire in Dana Point that killed an elderly woman?"
)

# FEWSHOT_1H_SCIDOCS = """Example 1:
# Document: Thermal Facial Analysis for Deception Detection Thermal imaging technology can be used to detect stress levels in humans based on the radiated heat from their face. In this paper, we use thermal imaging to monitor the periorbital region's thermal variations and test whether it can offer a discriminative signature for detecting deception. We start by presenting an overview on automated deception detection and propose a novel methodology, which we validate experimentally on 492 thermal responses (249 lies and 243 truths) extracted from 25 participants. The novelty of this paper lies in scoring a larger number of questions per subject, emphasizing a within-person approach for learning from data, proposing a framework for validating the decision making process, and correct evaluation of the generalization performance. A $k$ -nearest neighbor classifier was used to classify the thermal responses using different strategies for data representation. We report an 87% ability to predict the lie/truth responses based on a within-person methodology and fivefold cross validation. Our results also show that the between-person approach for modeling deception does not generalize very well across the training data.
# Title of the paper citing the document: """
# FEWSHOT_1A_SCIDOCS = "Analyzing Thermal and Visual Clues of Deception for a Non-Contact Deception Detection Approach"
# FEWSHOT_1H_SCIDOCS = """Example 1:
# Document: Multi-View Inpainting for Image-Based Scene Editing and Rendering We propose a method to remove objects such as people and cars from multi-view urban image datasets, enabling free-viewpoint IBR in the edited scenes. Our method combines information from multi-view 3D reconstruction with image inpainting techniques, by formulating the problem as an optimization of a global patch-based objective function. We use Image-Based Rendering (IBR) techniques to reproject information from neighboring views, and 3D multi-view stereo reconstruction to perform multiview coherent initialization for inpainting of pixels not filled by reprojection. Our algorithm performs multi-view consistent inpainting for color and 3D by blending reprojections with patch-based image inpainting. We run our algorithm on casually captured datasets, and Google StreetViewdata, removing objects cars, people and pillars, showing that our approach produces results of sufficient quality for free-viewpoint IBR on \"cleaned up\" scenes, as well as IBR scene editing, such as limited motion of real objects.
# Title of the paper citing the document: """
# FEWSHOT_1A_SCIDOCS = "Semantic-Aware Multi-View Inpainting for Enhanced Realism and Structure Consistency in Large-Scale Urban Scene Rendering"
FEWSHOT_1H_SCIDOCS = """Example 1:
Document: The emotional brain The discipline of affective neuroscience is concerned with the neural bases of emotion and mood. The past 30 years have witnessed an explosion of research in affective neuroscience that has addressed questions such as: which brain systems underlie emotions? How do differences in these systems relate to differences in the emotional experience of individuals? Do different regions underlie different emotions, or are all emotions a function of the same basic brain circuitry? How does emotion processing in the brain relate to bodily changes associated with emotion? And, how does emotion processing in the brain interact with cognition, motor behaviour, language and motivation?
Relevant Query: """
FEWSHOT_1H_SENT_SCIDOCS = """Example 1:
Document: The emotional brain The discipline of affective neuroscience is concerned with the neural bases of emotion and mood. The past 30 years have witnessed an explosion of research in affective neuroscience that has addressed questions such as: which brain systems underlie emotions? How do differences in these systems relate to differences in the emotional experience of individuals? Do different regions underlie different emotions, or are all emotions a function of the same basic brain circuitry? How does emotion processing in the brain relate to bodily changes associated with emotion? And, how does emotion processing in the brain interact with cognition, motor behaviour, language and motivation?
Sentence: And, how does emotion processing in the brain interact with cognition, motor behaviour, language and motivation?
Relevant Query: """
FEWSHOT_1A_SCIDOCS = "Emotion recognition in human-computer interaction"

FEWSHOT_1H_NQ = """Example 1:
Document: Mother's Day The United States celebrates Mother's Day on the second Sunday in May. In 1872 Julia Ward Howe called for women to join in support of disarmament and asked for 2 June 1872, to be established as a \"Mother's Day for Peace\". Her 1870 \"Appeal to womanhood throughout the world\" is sometimes referred to as Mother's Day Proclamation. But Howe's day was not for honouring mothers but for organizing pacifist mothers against war. In the 1880s and 1890s there were several further attempts to establish an American \"Mother's Day\", but these did not succeed beyond the local level.[129]
Relevant Query: """
FEWSHOT_1H_SENT_NQ = """Example 1:
Document: Mother's Day The United States celebrates Mother's Day on the second Sunday in May. In 1872 Julia Ward Howe called for women to join in support of disarmament and asked for 2 June 1872, to be established as a \"Mother's Day for Peace\". Her 1870 \"Appeal to womanhood throughout the world\" is sometimes referred to as Mother's Day Proclamation. But Howe's day was not for honouring mothers but for organizing pacifist mothers against war. In the 1880s and 1890s there were several further attempts to establish an American \"Mother's Day\", but these did not succeed beyond the local level.[129]
Sentence: In 1872 Julia Ward Howe called for women to join in support of disarmament and asked for 2 June 1872, to be established as a \"Mother's Day for Peace\".
Relevant Query: """
FEWSHOT_1A_NQ = "Who first proposed the idea of Mother's Day in the United States, and what was the original purpose behind it?"

FEWSHOT_1H_QUORA = """Example 1:
Question: How do you know if you are spiritually enlightened?
Duplicate Question: """
FEWSHOT_1H_SENT_QUORA = """Example 1:
Question: How do you know if you are spiritually enlightened?
Duplicate Question: """
FEWSHOT_1A_QUORA = (
    "What are the signs that indicate you have reached spiritual enlightenment?"
)

FEWSHOT_1H_NEWS = """Example 1:
Document: Title: Cleveland radio host loses her mind, accuses Browns\u2019 Jabrill Peppers and Joe Thomas of drug use Content: I\u2019m no radio-station manager or anything, but I do know this: Sports-talk hosts should not lob completely unfounded accusations of drug use by NFL players on the air. This just seems like common sense. Sabrina Parr of ESPN 850 in Cleveland did exactly that on Wednesday in talking about Jabrill Peppers, the Browns\u2019 first-round draft pick who was dinged for having a diluted urine sample at this year\u2019s NFL combine. To Parr, this means Peppers is abusing all sorts of drugs and is well on his way to becoming the next Josh Gordon, the Cleveland wide receiver who has been suspended multiple times for failed drug tests. \u201cHe\u2019s another Josh Gordon.\u2026 I\u2019ve seen it first-hand from a different vantage point, and it\u2019s the same thing all over again. How are you already high out of your mind, and you\u2019ve only been here for a week?\u201d Oh, and it\u2019s okay to say similar things about 10-time Pro Bowler Joe Thomas, who Parr joked is \u201con the lean, too,\u201d after he criticized the drug-testing procedures at the NFL combine. Suffice to say, the radio station fired Parr on Wednesday night. She later deleted a tweet about her quotes being taken out of context (oh, come on) but then seemed to subtweet Peppers on Thursday morning.
Relevant Article Headline:
"""
FEWSHOT_1A_NEWS = """Cleveland radio host fired after accusing Browns’ Jabrill Peppers and Joe Thomas of drug use"""

FEWSHOT_1H_DB = """Example 1:
Document: Coleophora tricolor The Basil-thyme case-bearer moth (Coleophora tricolor) is a moth of the Coleophoridae family. It is found in Great Britain, southern France and Greece.The wingspan is 14-18 mm.The larvae feed on Poaceae species, including Bromopsis erecta, Dactylis glomerata, Holcus lanatus, Koeleria macrantha, Phleum bertolonii and Poa pratensis. Young larvae eat the receptacle out of a floret of Acinos arvensis and uses the calyce as its first case. Even before the onset of winter it switches to grasses.
Relevant Query:
"""
FEWSHOT_1A_DB = """Basil-thyme case-bearer moth"""

FEWSHOT_1H_HOTPOT = """Example 1:
Document: Sukhbir Singh Gill Sukhbir Singh Gill (born 14 December 1975 in Chandigarh) is a former field hockey midfielder from India, who made his international debut for the Men's National Team in 1995 during the Sultan Azlan Shah Hockey Tournament in Kuala Lumpur, Malaysia. Gill represented his native country at the 2000 Summer Olympics in Sydney, Australia, where India finished in seventh place.
Relevant Query: """
FEWSHOT_1A_HOTPOT = "What position did Sukhbir Singh Gill play in field hockey?"


FEWSHOT_1H_MAP = {
    "fiqa": FEWSHOT_1H,
    "trec-covid": FEWSHOT_1H,
    "webis-touche2020": FEWSHOT_1H,
    "scifact": FEWSHOT_1H_SCIFACT,
    "nfcorpus": FEWSHOT_1H,
    "robust04": FEWSHOT_1H_ROBUST04,
    "arguana": FEWSHOT_1H_ARGUANA,
    "scidocs": FEWSHOT_1H_SCIDOCS,
    "nq": FEWSHOT_1H_NQ,
    "quora": FEWSHOT_1H_QUORA,
    "trec-news": FEWSHOT_1H_NEWS,
    "cqadupstack-android": FEWSHOT_1H,
    "cqadupstack-english": FEWSHOT_1H,
    "cqadupstack-gaming": FEWSHOT_1H,
    "cqadupstack-gis": FEWSHOT_1H,
    "cqadupstack-mathematica": FEWSHOT_1H,
    "cqadupstack-physics": FEWSHOT_1H,
    "cqadupstack-programmers": FEWSHOT_1H,
    "cqadupstack-stats": FEWSHOT_1H,
    "cqadupstack-tex": FEWSHOT_1H,
    "cqadupstack-unix": FEWSHOT_1H,
    "cqadupstack-webmasters": FEWSHOT_1H,
    "cqadupstack-wordpress": FEWSHOT_1H,
    "dbpedia-entity": FEWSHOT_1H_DB,
    "hotpotqa": FEWSHOT_1H_HOTPOT,
}
FEWSHOT_1H_SENT_MAP = {
    "fiqa": FEWSHOT_1H_SENT,
    "trec-covid": FEWSHOT_1H_SENT,
    "webis-touche2020": FEWSHOT_1H_SENT,
    "scifact": FEWSHOT_1H_SENT_SCIFACT,
    "nfcorpus": FEWSHOT_1H_SENT,
    "robust04": FEWSHOT_1H_SENT_ROBUST04,
    "arguana": FEWSHOT_1H_SENT_ARGUANA,
    "scidocs": FEWSHOT_1H_SENT_SCIDOCS,
    "nq": FEWSHOT_1H_SENT_NQ,
    "quora": FEWSHOT_1H_SENT_QUORA,
}
FEWSHOT_1A_MAP = {
    "fiqa": FEWSHOT_1A,
    "trec-covid": FEWSHOT_1A,
    "webis-touche2020": FEWSHOT_1A,
    "scifact": FEWSHOT_1A_SCIFACT,
    "nfcorpus": FEWSHOT_1A,
    "robust04": FEWSHOT_1A_ROBUST04,
    "arguana": FEWSHOT_1A_ARGUANA,
    "scidocs": FEWSHOT_1A_SCIDOCS,
    "nq": FEWSHOT_1A_NQ,
    "quora": FEWSHOT_1A_QUORA,
    "trec-news": FEWSHOT_1H_NEWS,
    "cqadupstack-android": FEWSHOT_1A,
    "cqadupstack-english": FEWSHOT_1A,
    "cqadupstack-gaming": FEWSHOT_1A,
    "cqadupstack-gis": FEWSHOT_1A,
    "cqadupstack-mathematica": FEWSHOT_1A,
    "cqadupstack-physics": FEWSHOT_1A,
    "cqadupstack-programmers": FEWSHOT_1A,
    "cqadupstack-stats": FEWSHOT_1A,
    "cqadupstack-tex": FEWSHOT_1A,
    "cqadupstack-unix": FEWSHOT_1A,
    "cqadupstack-webmasters": FEWSHOT_1A,
    "cqadupstack-wordpress": FEWSHOT_1A,
    "dbpedia-entity": FEWSHOT_1A_DB,
    "hotpotqa": FEWSHOT_1A_HOTPOT,
}

FEWSHOT_2H_DOC = """Example 2:
Document: Yes, there is a lot they are leaving out, and I would be extremely skeptical of them because of the ""reasons"" they give for being able to charge $0 commissions. Their reasons are that they don\'t have physical locations and high overhead costs, the reality is that they are burning venture capital on exchange fees until they actually start charging everyone they suckered into opening accounts. They also get paid by exchanges when users provide liquidity. These are called trade rebates in the maker-taker model. They will start offering margin accounts and charging interest. They are [likely] selling trade data to high frequency trading firms that then fill your stock trades at worse prices (Robinhood users are notorious for complaining about the fills). They may well be able to keep commissions low, as that has been a race to the bottom for a long time. But if they were doing their users any actual favors, then they would be also paying users the rebates that exchanges pay them for liquidity. Robinhood isn\'t doing anything unique as all brokers do what I mentioned along with charging commissions, and it is actually amazing their sales pitch ""$0 commissions because we are just a mobile app lol"" was enough for their customers. They are just being disingenuous."""
FEWSHOT_2H = f"""{FEWSHOT_2H_DOC}
Relevant Query: """
FEWSHOT_2H_SENT = f"""{FEWSHOT_2H_DOC}
Sentence: Yes, there is a lot they are leaving out, and I would be extremely skeptical of them because of the ""reasons"" they give for being able to charge $0 commissions.
Relevant Query: """
FEWSHOT_2H_KW = f"""{FEWSHOT_2H_DOC}
Keywords: 1.Robinhood stock broker 2.Trade rebates 3.High frequency trading firms
Relevant Query: """
FEWSHOT_2H_KWGEN = f"""{FEWSHOT_2H_DOC}
Keywords: """
FEWSHOT_2H_SM = f"""{FEWSHOT_2H_DOC}
Summary: The document expresses skepticism toward a brokerage firm's claim of offering zero-commission trading due to low overhead costs. The author argues that the firm compensates for this by using venture capital to cover exchange fees, earning trade rebates in the maker-taker model, charging interest on margin accounts, and potentially selling trade data to high-frequency trading firms, which may result in less favorable trade executions for users. The critique suggests that the firm's practices are not unique among brokers and labels their marketing approach as disingenuous.
Relevant Query: """
FEWSHOT_2H_SUMMARY = """Example 2:
Summary: The document expresses skepticism about brokers like Robinhood offering $0 commissions, suggesting that the claim of low overhead costs is misleading. Instead, the broker is likely subsidizing commissions with venture capital and profiting through other means, such as trade rebates from exchanges in the maker-taker model, potentially selling trade data to high-frequency traders, and planning to charge interest on margin accounts. While commission fees may remain low, the author argues that Robinhood, like other brokers, isn’t passing trade rebates to users, calling the “$0 commission” marketing approach disingenuous.
Relevant Query: """
FEWSHOT_2H_STREAMLINED = """Example 2:
Clarified Document: Be cautious about brokers offering "$0 commissions." Although they claim it’s possible due to low overhead costs, the reality is that they're likely covering expenses with venture capital for now and could start charging fees later. They also earn trade rebates from exchanges when users provide liquidity, and they might sell trade data to high-frequency trading firms, which can lead to less favorable trade prices for users. Brokers like Robinhood may keep commissions low, but they’re not passing these rebates on to users, making their “$0 commissions” marketing somewhat misleading.
Relevant Query: """
FEWSHOT_2A = "How does Robinhood stock broker make money?"
FEWSHOT_2A_SUMMARY = "Why is Robinhood's $0 commission model considered misleading, and how does the company actually make money?"
FEWSHOT_2A_STREAMLINED = (
    "Why should investors be cautious of brokers offering $0 commissions?"
)
FEWSHOT_2A_KWGEN = (
    "1.Robinhood stock broker 2.Trade rebates 3.High frequency trading firms"
)

# FEWSHOT_2H_TRECCOVID = """Example 2:
# Document: Benefits and drawbacks of SILS cholecystectomy: a report of 60 SILS cholecystectomies with conventional instrumentation from an academic center. BACKGROUND Single-incision laparoscopic surgery is a rapidly emerging approach to gallbladder disease. METHODS From February 2009 to September 2010, 60 patients were subjected to single-incision laparoscopic cholecystectomy. In all the patients, a 12-mm incision was made in the umbilicus and a 2-trocar technique was applied. Gallbladder was suspended with 2 sutures and the procedure was accomplished with standard partly reusable laparoscopic instruments. RESULTS In all, 41 women (68.3%) and 19 men (31.7%) were enrolled in this study. Mean age was 50.7 years (range = 17-72 years), mean body mass index was 26.2 kg/m(2) (range = 18.3-37.7 kg/m(2)) and mean operative time was 52.6 minutes (range = 30-120 minutes). No mortality or morbidity was recorded and hospital stay was less than 24 hours. At follow-up visits, no complications were recorded and cosmesis was excellent. CONCLUSION Single-incision laparoscopic cholecystectomy can be safely performed with conventional instrumentation with minimal cost.
# Scientific Query: """
# FEWSHOT_2A_TRECCOVID = "What are the safety, cosmetic outcomes, and cost-effectiveness of single-incision laparoscopic cholecystectomy (SILS) using conventional instruments?"

# FEWSHOT_2H_TOUCHE = """Example 2:
# Document: Cheerleaders are good for pro sports Let's take a look at that 78%, shall we? My guess, which i'm quite sure is accurate, is that about 66% of the people who favor cheerleaders are slobbering, woman-demeaning men, who enjoy their female counterparts best while in scantily-clad outfits dancing around, or just plain nude in a $2.99 per minute porno. About 11% percent are girls ranged 4-17 who are cheerleaders themselves, and 1% are stepford wives who seem to enjoy anything placed in front of them. May I ask where you received this statistic from? Focusing less on the 78 percent, lets take a look at the 22 percent. I can assure you that these 22% don't just not enjoy cheerleaders, but dislike them greatly (hate is such an extreme word). The 22% is made up of 15% girls who have a few bones of sense in their body, and have better, more intelligent things to do than prance around in a demeaning suit, cheering on their show-stopping men counterparts who seem to allot to so much more than they ever will, 6% is made up by grouchy old men and women who wish to put a pox on cheerleading, and, most importantly, 1% of the 22% of people who despise cheerleaders are teenage boys like me who can actually hold a grasp on their raging hormones for five minutes, and enjoy why you went to the basketball or football game; to watch the wonders of sports. Oh, and contrary to popular belief, it's the players giving it their all, spectacular plays, and tight games that rally crowds. Not cheerleaders.
# Relevant Query: """
# FEWSHOT_2A_TOUCHE = "What are the criticisms of cheerleading in professional sports and its impact on the fan experience?"

FEWSHOT_2H_SCIFACT = """Example 2:
Document: Replication Fork Stability Confers Chemoresistance in BRCA-deficient Cells Cells deficient in the Brca1 and Brca2 genes have reduced capacity to repair DNA double-strand breaks by homologous recombination and consequently are hypersensitive to DNA-damaging agents, including cisplatin and poly(ADP-ribose) polymerase (PARP) inhibitors. Here we show that loss of the MLL3/4 complex protein, PTIP, protects Brca1/2-deficient cells from DNA damage and rescues the lethality of Brca2-deficient embryonic stem cells. However, PTIP deficiency does not restore homologous recombination activity at double-strand breaks. Instead, its absence inhibits the recruitment of the MRE11 nuclease to stalled replication forks, which in turn protects nascent DNA strands from extensive degradation. More generally, acquisition of PARP inhibitors and cisplatin resistance is associated with replication fork protection in Brca2-deficient tumour cells that do not develop Brca2 reversion mutations. Disruption of multiple proteins, including PARP1 and CHD4, leads to the same end point of replication fork protection, highlighting the complexities by which tumour cells evade chemotherapeutic interventions and acquire drug resistance.
Claim Supported By Document: """
FEWSHOT_2H_SENT_SCIFACT = """Example 2:
Document: Replication Fork Stability Confers Chemoresistance in BRCA-deficient Cells Cells deficient in the Brca1 and Brca2 genes have reduced capacity to repair DNA double-strand breaks by homologous recombination and consequently are hypersensitive to DNA-damaging agents, including cisplatin and poly(ADP-ribose) polymerase (PARP) inhibitors. Here we show that loss of the MLL3/4 complex protein, PTIP, protects Brca1/2-deficient cells from DNA damage and rescues the lethality of Brca2-deficient embryonic stem cells. However, PTIP deficiency does not restore homologous recombination activity at double-strand breaks. Instead, its absence inhibits the recruitment of the MRE11 nuclease to stalled replication forks, which in turn protects nascent DNA strands from extensive degradation. More generally, acquisition of PARP inhibitors and cisplatin resistance is associated with replication fork protection in Brca2-deficient tumour cells that do not develop Brca2 reversion mutations. Disruption of multiple proteins, including PARP1 and CHD4, leads to the same end point of replication fork protection, highlighting the complexities by which tumour cells evade chemotherapeutic interventions and acquire drug resistance.
Sentence: Cells deficient in the Brca1 and Brca2 genes have reduced capacity to repair DNA double-strand breaks by homologous recombination and consequently are hypersensitive to DNA-damaging agents, including cisplatin and poly(ADP-ribose) polymerase (PARP) inhibitors.
Claim Supported By Document: """
FEWSHOT_2A_SCIFACT = "Loss of PTIP confers chemoresistance in BRCA-deficient cells by stabilizing replication forks and preventing DNA degradation, independent of homologous recombination restoration."

# FEWSHOT_2H_NFCORPUS = """Example 2:
# Document: Total antioxidant capacity of diet and risk of stroke: a population-based prospective cohort of women. BACKGROUND AND PURPOSE: Consumption of antioxidant-rich foods may reduce the risk of stroke by inhibition of oxidative stress and inflammation. Total antioxidant capacity (TAC) takes into account all antioxidants and the synergistic effects between them. We examined the association between dietary TAC and stroke incidence in cardiovascular disease (CVD)-free women and in women with CVD history at baseline. METHODS: The study included women (31,035 CVD-free and 5680 with CVD history at baseline), aged 49 to 83 years, from the Swedish Mammography Cohort. Diet was assessed with a food frequency questionnaire. Dietary TAC was calculated using oxygen radical absorbance capacity values. Stroke cases were ascertained by linkage with the Swedish Hospital Discharge Registry. RESULTS: During follow-up (September 1997 to December 2009), we identified 1322 stroke cases (988 cerebral infarctions, 226 hemorrhagic strokes, and 108 unspecified strokes) among CVD-free women and 1007 stroke cases (796 cerebral infarctions, 100 hemorrhagic strokes, and 111 unspecified strokes) among women with a CVD history. The multivariable hazard ratio of total stroke comparing the highest with the lowest quintile of dietary TAC was 0.83 (95% CI, 0.70-0.99; P for trend=0.04) in CVD-free women. Among women with a CVD history, the hazard ratios for the highest versus lowest quartile of TAC were 0.90 (95% CI, 0.75-1.07; P for trend=0.30) for total stroke and 0.55 (95% CI, 0.32-0.95; P for trend=0.03) for hemorrhagic stroke. CONCLUSIONS: These findings suggest that dietary TAC is inversely associated with total stroke among CVD-free women and hemorrhagic stroke among women with CVD history.
# Scientific Query: """
# FEWSHOT_2A_NFCORPUS = "Is there an association between dietary total antioxidant capacity (TAC) and reduced stroke risk in women with or without a history of cardiovascular disease?"

FEWSHOT_2H_ARGUANA = """Example 2:
Document: Within cities land grabbing is a myth. A number of cases shown as political land-grabbing and rent-seeking are misrepresented, and misunderstood. Difficulties remain in defining what is a land grab and the extent of which the state, and politics, are involved in land speculations.  The media coverage of evictions in Mogadishu showcase the myth and hyperbole surrounding African politics and evictions. The government are entitled to reclaim land and reform it for public use [1] .  [1] See BBC News (2013) for full debate, whereby Mohammed Yusuf, an Official at Mogadishu City, defends the eviction.
Counter Argument:
"""
FEWSHOT_2H_SENT_ARGUANA = """Example 2:
Document: Within cities land grabbing is a myth. A number of cases shown as political land-grabbing and rent-seeking are misrepresented, and misunderstood. Difficulties remain in defining what is a land grab and the extent of which the state, and politics, are involved in land speculations.  The media coverage of evictions in Mogadishu showcase the myth and hyperbole surrounding African politics and evictions. The government are entitled to reclaim land and reform it for public use [1] .  [1] See BBC News (2013) for full debate, whereby Mohammed Yusuf, an Official at Mogadishu City, defends the eviction.
Sentence: The media coverage of evictions in Mogadishu showcase the myth and hyperbole surrounding African politics and evictions.
Counter Argument:
"""
FEWSHOT_2A_ARGUANA = "Land grabbing is a significant problem in urban areas, where governments and political figures often exploit land for personal or political gain, leading to forced evictions and the displacement of vulnerable communities. Media coverage of these evictions highlights corruption and the misuse of power, showcasing how political agendas can infringe upon citizens' rights to secure housing."

FEWSHOT_2H_ROBUST04 = """Example 2:
Document: 930818\nA LABOUR government would impose a levy of up to 1.5 per cent of payroll\ncosts on companies which failed to comply with training guidelines, Mr\nGordon Brown, shadow chancellor, said yesterday.\nThe levy, intended to help pay for upgrading government training programmes,\ncompares with earlier plans for a maximum levy of 0.5 per cent on all\ncompanies not spending that amount.\nThe revised proposal emerged in a paper for Labour's annual conference next\nmonth, in which Mr Brown further distances the party from the\nhigher-taxation manifesto on which it fought the 1992 general election.\nPromising to cut taxes 'if I can', Mr Brown confirmed the Labour\nleadership's determination to discard the party's redistributionist image.\n'Labour is not against wealth, nor will we seek to penalise it,' he said.\nMr Brown said the revised training proposals were aimed at encouraging\ncompanies to develop their own training programmes, rather than rely on the\ngovernment.\n'There are a large number of companies which are failing to make the\ntraining investment which is necessary. That is not only harming the country\nas a whole, it is harming those companies which are prepared to make the\ninvestment because they are finding that their trained workers are being\npoached,' he said.\nThe revised proposal is based on similar schemes operating in France,\nAustralia and New Zealand. Labour officials are believed to have concluded\nthat the amount raised through the original scheme would have been\ninsufficient to finance a worthwhile training programme.\nThe proposal was dismissed as a 'distraction' by the Confederation of\nBritish Industry, which said spending on training had been rising since\n1989, despite the recession.\n'Government regulation of this kind would just lead to a reclassification of\nexisting activities as companies tried to comply with the rules,' said Mr\nTony Webb, CBI training director. 'What we need is a focus on increasing\ntraining through cultural change, not an artificial scheme requiring a set\nlevel of spending.'\nMr Brown presented the training levy as a key component of Labour's revised\neconomic strategy, unveiled last month, which focuses on increasing\ninvestment and competition.\nMr Michael Portillo, the chief secretary to the treasury, said Labour's\npolicy document was full of 'platitudinous' rubbish. 'It is just silly,\nparticularly for the Labour party, to go round talking about tax cuts at the\nmoment,' he said.\nLabour's Economic Approach, 1993 conference paper. 150 Walworth Road, London\nSE1. (071 701 1234).", "title": "FT 18 AUG 93 / Labour levy would enforce training guidelines
Relevant Query: """
FEWSHOT_2H_SENT_ROBUST04 = """Example 2:
Document: 930818\nA LABOUR government would impose a levy of up to 1.5 per cent of payroll\ncosts on companies which failed to comply with training guidelines, Mr\nGordon Brown, shadow chancellor, said yesterday.\nThe levy, intended to help pay for upgrading government training programmes,\ncompares with earlier plans for a maximum levy of 0.5 per cent on all\ncompanies not spending that amount.\nThe revised proposal emerged in a paper for Labour's annual conference next\nmonth, in which Mr Brown further distances the party from the\nhigher-taxation manifesto on which it fought the 1992 general election.\nPromising to cut taxes 'if I can', Mr Brown confirmed the Labour\nleadership's determination to discard the party's redistributionist image.\n'Labour is not against wealth, nor will we seek to penalise it,' he said.\nMr Brown said the revised training proposals were aimed at encouraging\ncompanies to develop their own training programmes, rather than rely on the\ngovernment.\n'There are a large number of companies which are failing to make the\ntraining investment which is necessary. That is not only harming the country\nas a whole, it is harming those companies which are prepared to make the\ninvestment because they are finding that their trained workers are being\npoached,' he said.\nThe revised proposal is based on similar schemes operating in France,\nAustralia and New Zealand. Labour officials are believed to have concluded\nthat the amount raised through the original scheme would have been\ninsufficient to finance a worthwhile training programme.\nThe proposal was dismissed as a 'distraction' by the Confederation of\nBritish Industry, which said spending on training had been rising since\n1989, despite the recession.\n'Government regulation of this kind would just lead to a reclassification of\nexisting activities as companies tried to comply with the rules,' said Mr\nTony Webb, CBI training director. 'What we need is a focus on increasing\ntraining through cultural change, not an artificial scheme requiring a set\nlevel of spending.'\nMr Brown presented the training levy as a key component of Labour's revised\neconomic strategy, unveiled last month, which focuses on increasing\ninvestment and competition.\nMr Michael Portillo, the chief secretary to the treasury, said Labour's\npolicy document was full of 'platitudinous' rubbish. 'It is just silly,\nparticularly for the Labour party, to go round talking about tax cuts at the\nmoment,' he said.\nLabour's Economic Approach, 1993 conference paper. 150 Walworth Road, London\nSE1. (071 701 1234).", "title": "FT 18 AUG 93 / Labour levy would enforce training guidelines
Sentence: The revised training proposals were aimed at encouraging companies to develop their own training programmes, rather than rely on the government.
Relevant Query: """
FEWSHOT_2A_ROBUST04 = "What are Labour's proposed training levy guidelines for companies and the rationale behind them?"

# FEWSHOT_2H_SCIDOCS = """Example 2:
# Document: Vision-Based Gesture Recognition: A Review The use of gesture as a natural interface serves as a motivating force for research in modeling, analyzing and recognition of gestures. In particular, human computer intelligent interaction needs vision-based gesture recognition, which involves many interdisciplinary studies. A survey on recent vision-based gesture recognition approaches is given in this paper. We shall review methods of static hand posture and temporal gesture recognition. Several application systems of gesture recognition are also described in this paper. We conclude with some thoughts about future research directions.
# Title of the paper citing the document: """
# FEWSHOT_2A_SCIDOCS = "ArSLAT: Arabic Sign Language Alphabets Translator"
# FEWSHOT_2H_SCIDOCS = """Example 2:
# Document: Practical Privacy-Preserving Medical Diagnosis Using Homomorphic Encryption The use of remote services offered by cloud providers have been popular in the last lustrum. Services allow users to store remote files, or to analyze data for several purposes, like health-care or message analysis. However, when personal data are sent to the Cloud, users may lose privacy on the data-content, and on the other side cloud providers may use those data for their own businesses. In this paper, we present our solution to analyze users health-data directly into the Cloud while preserving users privacy. Our solution makes use of homomorphic encryption to protect users data during the analysis. In particular, we developed a mobile application that offloads users data into the Cloud, and a homomorphic encryption algorithm that processes those data without leaking any information to the Cloud provider. Performed empirical tests show that our HE algorithm is able to evaluate users data in reasonable time proving the feasibility of this emerging way of private-data evaluation.
# Title of the paper citing the document: """
# FEWSHOT_2A_SCIDOCS = "Optimizing Homomorphic Encryption for Real-Time Privacy-Preserving Medical Diagnostics on Mobile Platforms: Enhancing Feasibility and Efficiency in Cloud-Based Health Data Analysis"
FEWSHOT_2H_SCIDOCS = """Example 2:
Document: Fast exact string matching algorithms String matching is the problem of finding all the occurrences of a pattern in a text. We propose a very fast new family of string matching algorithms based on hashing q-grams. The new algorithms are the fastest on many cases, in particular, on small size alphabets. \u00a9 2007 Elsevier B.V. All rights reserved.
Relevant Query: """
FEWSHOT_2H_SENT_SCIDOCS = """Example 2:
Document: Fast exact string matching algorithms String matching is the problem of finding all the occurrences of a pattern in a text. We propose a very fast new family of string matching algorithms based on hashing q-grams. The new algorithms are the fastest on many cases, in particular, on small size alphabets. \u00a9 2007 Elsevier B.V. All rights reserved.
Sentence: String matching is the problem of finding all the occurrences of a pattern in a text.
Relevant Query: """
FEWSHOT_2A_SCIDOCS = "Text matching algorithms for sentiment analysis in social media"

FEWSHOT_2H_NQ = """Example 2:
Document: Gold mining in Alaska The 1886 discovery of gold on Franklin's Bar on the Fortymile River touched off Interior Alaska's first gold rush. The mining boom ushered in a wave of settlement that forever changed the place, not only for its new residents but for the Athabascan Indians who occupied this region long before them. The miners who prospected nearly every creek in the region eventually extracted more than a half-million ounces of gold from the Fortymile, including a 56.8 troy ounce nugget, Alaska's 15th-largest.[32] Reports of starvation and lawlessness among the miners resulted in the Army sending troops to the Eagle area to provide law enforcement in 1899. Soldiers soon began work on a trail from Valdez to Eagle.[33]
Relevant Query: """
FEWSHOT_2H_SENT_NQ = """Example 2:
Document: Gold mining in Alaska The 1886 discovery of gold on Franklin's Bar on the Fortymile River touched off Interior Alaska's first gold rush. The mining boom ushered in a wave of settlement that forever changed the place, not only for its new residents but for the Athabascan Indians who occupied this region long before them. The miners who prospected nearly every creek in the region eventually extracted more than a half-million ounces of gold from the Fortymile, including a 56.8 troy ounce nugget, Alaska's 15th-largest.[32] Reports of starvation and lawlessness among the miners resulted in the Army sending troops to the Eagle area to provide law enforcement in 1899. Soldiers soon began work on a trail from Valdez to Eagle.[33]
Sentence: The 1886 discovery of gold on Franklin's Bar on the Fortymile River touched off Interior Alaska's first gold rush.
Relevant Query: """
FEWSHOT_2A_NQ = "What sparked the first gold rush in Interior Alaska, and how did it affect the region and its people?"

FEWSHOT_2H_QUORA = """Example 2:
Question: How can I learn about the basics of computer and information security?
Duplicate Question: """
FEWSHOT_2A_QUORA = "What is the best way to get started with the fundamentals of cybersecurity and information protection?"

FEWSHOT_2H_NEWS = """Example 2:
Document: Title: Government marijuana looks nothing like the real stuff. See for yourself. Content: Take a look at the photo above. That's what most marijuana consumers picture when they think \u201cmarijuana\u201d \u2014 chunks of pungent green plant material coated in sticky, crystallized THC-rich resin. To investigate the real-world effects of marijuana, however, researchers need a product that looks and feels like the real thing. And they're increasingly frustrated with government weed that is something else entirely. Here they are side by side: \u201cIn two decades of smoking weed, I've never seen anything that looks like that,\u201d Browne said. \u201cPeople typically smoke the flower of the plant, but here you can clearly see stems and leaves in there as well, parts that should be discarded. Inhaling that would be like eating an apple, including the seeds inside it and the branch it grew on.\u201d It's unclear if this is an exceptionally bad batch, but there's reason to strongly suspect it's typical of what most researchers are given. The problems with the Mississippi weed go well beyond aesthetics. For a researcher, it's difficult to assess the real-world impact of high-end pot if you only have access to the low-quality stuff. It's akin to investigating the effects of bourbon by giving people Bud Light. For certain types of research this isn't necessarily a problem, says Rick Doblin, founding director of the Multidisciplinary Association for Psychedelic Studies, a group that's been working with Sisley on the PTSD trial. \"[NIDA's] marijuana is fine if you want to do academic research,\u201d Doblin said \u2014 studies that look at how marijuana affects the body in a laboratory setting, for instance. But NIDA's weed doesn't pass muster if you want to know how marijuana use is affecting people in the real world. Or if you want to run highly controlled medical experiments, like the one Sisley and Doblin are working on. It's not even tested for some common contaminants, like yeast and mold, that many states now check for as part of their regulatory regimes. Doblin said the marijuana they received from NIDA showed levels of mold and yeast that far exceeded standards for some states, like Colorado and Washington. Be they opted to go ahead with the trial since additional testing confirmed that none of the strains of mold and yeast found in the plant material posed a risk to humans.
Relevant Article Headline:
"""
FEWSHOT_2A_NEWS = """Researchers say federal research marijuana is low-quality and unrepresentative of real-world cannabis, complicating clinical trials"""

FEWSHOT_2H_DB = """Example 2:
Document: Ch\u00e2teau de Rayne-Vigneau Ch\u00e2teau de Rayne-Vigneau is a sweet white wine ranked as Premier Cru Class\u00e9 (French, \u201cFirst Growth\u201d) in the original Bordeaux Wine Official Classification of 1855. Belonging to the Sauternes appellation in Gironde, in the region of Graves, the winery is located in Bommes. It has been owned by Cr\u00e9dit Agricole since 2004.
Relevant Query:
"""
FEWSHOT_2A_DB = """Which Sauternes Premier Cru Classé estate is owned by Crédit Agricole?"""

FEWSHOT_2H_HOTPOT = """Example 2:
Document: 2011\u201312 Bobsleigh World Cup The 2011\u201312 Bobsleigh World Cup is a multi race tournament over a season for bobsleigh. The season started on 2 December 2011 in Igls, Austria and ended on 11 February 2012 in Calgary, Alberta, Canada. The World Cup is organised by the FIBT who also run World Cups and Championships in skeleton. This season iss sponsored by Viessmann.
Relevant Query: """
FEWSHOT_2A_HOTPOT = "Which city hosted the final event of the 2011–12 Bobsleigh World Cup season?"

FEWSHOT_2H_MAP = {
    "fiqa": FEWSHOT_2H,
    "trec-covid": FEWSHOT_2H,
    "webis-touche2020": FEWSHOT_2H,
    "scifact": FEWSHOT_2H_SCIFACT,
    "nfcorpus": FEWSHOT_2H,
    "robust04": FEWSHOT_2H_ROBUST04,
    "arguana": FEWSHOT_2H_ARGUANA,
    "scidocs": FEWSHOT_2H_SCIDOCS,
    "nq": FEWSHOT_2H_NQ,
    "quora": FEWSHOT_2H_QUORA,
    "trec-news": FEWSHOT_2H_NEWS,
    "cqadupstack-android": FEWSHOT_2H,
    "cqadupstack-english": FEWSHOT_2H,
    "cqadupstack-gaming": FEWSHOT_2H,
    "cqadupstack-gis": FEWSHOT_2H,
    "cqadupstack-mathematica": FEWSHOT_2H,
    "cqadupstack-physics": FEWSHOT_2H,
    "cqadupstack-programmers": FEWSHOT_2H,
    "cqadupstack-stats": FEWSHOT_2H,
    "cqadupstack-tex": FEWSHOT_2H,
    "cqadupstack-unix": FEWSHOT_2H,
    "cqadupstack-webmasters": FEWSHOT_2H,
    "cqadupstack-wordpress": FEWSHOT_2H,
    "dbpedia-entity": FEWSHOT_2H_DB,
    "hotpotqa": FEWSHOT_2H_HOTPOT,
}

FEWSHOT_2H_SENT_MAP = {
    "fiqa": FEWSHOT_2H_SENT,
    "trec-covid": FEWSHOT_2H_SENT,
    "webis-touche2020": FEWSHOT_2H_SENT,
    "scifact": FEWSHOT_2H_SENT_SCIFACT,
    "nfcorpus": FEWSHOT_2H_SENT,
    "robust04": FEWSHOT_2H_SENT_ROBUST04,
    "arguana": FEWSHOT_2H_SENT_ARGUANA,
    "scidocs": FEWSHOT_2H_SENT_SCIDOCS,
    "nq": FEWSHOT_2H_SENT_NQ,
    "quora": FEWSHOT_2H_QUORA,
}
FEWSHOT_2A_MAP = {
    "fiqa": FEWSHOT_2A,
    "trec-covid": FEWSHOT_2A,
    "webis-touche2020": FEWSHOT_2A,
    "scifact": FEWSHOT_2A_SCIFACT,
    "nfcorpus": FEWSHOT_2A,
    "robust04": FEWSHOT_2A_ROBUST04,
    "arguana": FEWSHOT_2A_ARGUANA,
    "scidocs": FEWSHOT_2A_SCIDOCS,
    "nq": FEWSHOT_2A_NQ,
    "quora": FEWSHOT_2A_QUORA,
    "trec-news": FEWSHOT_2A_NEWS,
    "cqadupstack-android": FEWSHOT_2A,
    "cqadupstack-english": FEWSHOT_2A,
    "cqadupstack-gaming": FEWSHOT_2A,
    "cqadupstack-gis": FEWSHOT_2A,
    "cqadupstack-mathematica": FEWSHOT_2A,
    "cqadupstack-physics": FEWSHOT_2A,
    "cqadupstack-programmers": FEWSHOT_2A,
    "cqadupstack-stats": FEWSHOT_2A,
    "cqadupstack-tex": FEWSHOT_2A,
    "cqadupstack-unix": FEWSHOT_2A,
    "cqadupstack-webmasters": FEWSHOT_2A,
    "cqadupstack-wordpress": FEWSHOT_2A,
    "dbpedia-entity": FEWSHOT_2A_DB,
    "hotpotqa": FEWSHOT_2A_HOTPOT,
}

FEWSHOT_3H_DOC = """Example 3:
Document: You could, but the bank won't let you... If you're a sole proprietor - then you could probably open a personal account and just use it, and never tell them that is actually a business. However, depending on your volume of operations, they may switch you on their own to business account by the pattern of your transactions. For corporations, you cannot use a personal account since the corporation is a separate legal entity that owns the funds. Also, you're generally required to separate corporate and personal funds to keep the limited liability protection (which is why you have the corporation to begin with). Generally, business accounts have much higher volumes and much more transactions than personal accounts, and it costs more for the banks to run them. In the US, some banks offer free, or very low-cost, business accounts for small businesses that don't need too many transactions. I'm sure if you shop around, you'll find those in Canada as well."""
FEWSHOT_3H = f"""{FEWSHOT_3H_DOC}
Relevant Query: """
FEWSHOT_3H_SENT = f"""{FEWSHOT_3H_DOC}
Sentence: You could, but the bank won't let you... If you're a sole proprietor - then you could probably open a personal account and just use it, and never tell them that is actually a business.
Relevant Query: """
FEWSHOT_3H_KW = f"""{FEWSHOT_3H_DOC}
Keywords: 1.Sole proprietor 2.Corporation 3.Business account
Relevant Query: """
FEWSHOT_3H_KWGEN = f"""{FEWSHOT_3H_DOC}
Keywords: """
FEWSHOT_3H_SM = f"""{FEWSHOT_3H_DOC}
Summary: The document discusses the feasibility of using a personal bank account for business purposes. For sole proprietors, it's possible to use a personal account without disclosing its business use, but high transaction volumes may prompt the bank to reclassify it as a business account. For corporations, using a personal account is not permissible because the corporation is a separate legal entity, and mixing funds can jeopardize limited liability protection. While business accounts often involve higher costs due to increased transactions, some banks offer free or low-cost options for small businesses with fewer transactions, both in the U.S. and potentially in Canada.
Relevant Query: """
FEWSHOT_3H_SUMMARY = """Example 3:
Summary: The document explains that while sole proprietors might use a personal bank account for business, banks may reclassify it as a business account based on transaction patterns. Corporations, however, must use business accounts to separate corporate funds from personal funds, preserving limited liability protection. Business accounts generally handle higher transaction volumes, which increases bank costs. In the U.S., some banks offer low-cost business accounts for small businesses, and similar options are likely available in Canada.
Relevant Query: """
FEWSHOT_3H_STREAMLINED = """Example 3:
Clarified Document: If you're a sole proprietor, you might be able to use a personal bank account for business, but banks may switch it to a business account based on your transaction volume. For corporations, a business account is required since the corporation is a separate legal entity, and separating corporate and personal funds is essential to maintain limited liability. Business accounts generally handle more transactions, so they cost banks more to operate. In the U.S., some banks offer free or low-cost business accounts for small businesses with fewer transactions, and similar options are likely available in Canada.
Relevant Query: """
FEWSHOT_3A = "Why do banks require small businesses to open a business bank account instead of a cheaper personal one?"
FEWSHOT_3A_SUMMARY = "Why must corporations use separate business bank accounts to maintain limited liability protection?"
FEWSHOT_3A_STREAMLINED = "Why might banks reclassify a sole proprietor's personal bank account as a business account based on transaction volume?"
FEWSHOT_3A_KWGEN = "1.Sole proprietor 2.Corporation 3.Business account"

# FEWSHOT_3H_TRECCOVID = """Example 3:
# Document: Deletion of both the Tyrosine-Based Endocytosis Signal and the Endoplasmic Reticulum Retrieval Signal in the Cytoplasmic Tail of Spike Protein Attenuates Porcine Epidemic Diarrhea Virus in Pigs Porcine epidemic diarrhea virus (PEDV) causes high mortality in neonatal piglets. The PEDV spike (S) protein contains two intracellular sorting motifs, Yxx\u03a6 (tyrosine-based motif YEVF or YEAF) and KVHVQ at the cytoplasmic tail, yet their functions have not been fully elucidated. Some Vero cell-adapted and/or attenuated PEDV variants contain ablations in these two motifs. We hypothesized that these motifs contribute to viral pathogenicity. By transiently expressing PEDV S proteins with mutations in the motifs, we confirmed that the motif KVHVQ is involved in retention of the S proteins in the endoplasmic reticulum (ER)-Golgi intermediate compartment (ERGIC). In addition, we showed that the Yxx\u03a6 motif triggers endocytosis of S proteins. These two motifs synergistically regulate the level of S expressed on the cell surface. To investigate their role in viral pathogenicity, we generated three recombinant PEDVs by introducing deletions or a mutation in the two motifs of the infectious clone of PEDV PC22A strain (icPC22A): (i) ic\u039410aa (\u0394Yxx\u03a6EKVHVQ), (ii) ic\u03945aa (\u0394KVHVQ), and (iii) icYA (Y1378A, to an inactivated motif, AEVF). Infection of Vero cells with ic\u039410aa resulted in larger syncytia and more virions, with reduced numbers of S protein projections on the surface compared with icPC22A. Furthermore, we orally inoculated five groups of 5-day-old gnotobiotic piglets with the three mutants, icPC22A, or a mock treatment. Mutant ic\u039410aa caused less severe diarrhea rate and significantly milder intestinal lesions than icPC22A, ic\u03945aa, and icYA. These data suggest that the deletion of both motifs can reduce the virulence of PEDV in piglets. IMPORTANCE Many coronaviruses (CoVs) possess conserved motifs Yxx\u03a6 and/or KxHxx/KKxx in the cytoplasmic tail of the S protein. The KxHxx/KKxx motif has been identified as the ER retrieval signal, but the function of the Yxx\u03a6 motif in the intracellular sorting of CoV S proteins remains controversial. In this study, we showed that the Yxx\u03a6 of PEDV S protein is an endocytosis signal. Furthermore, using reverse genetics technology, we evaluated its role in PEDV pathogenicity in neonatal piglets. Our results explain one attenuation mechanism of Vero cell-adapted PEDV variants lacking functional Yxx\u03a6 and KVHVQ motifs. Knowledge from this study may aid in the design of efficacious live attenuated vaccines against PEDV, as well as other CoVs bearing the same motif in their S protein.
# Scientific Query: """
# FEWSHOT_3A_TRECCOVID = "How do mutations in the YxxΦ and KVHVQ motifs of the spike protein affect the virulence and pathogenicity of Porcine Epidemic Diarrhea Virus (PEDV) in neonatal piglets?"

# FEWSHOT_3H_TOUCHE = """Example 3:
# Document: Sarah Palin should be elected as \"Queen of Earth\" Defense 1My opponent fails. He fails to see that I was stating a figure of speech in my statment about \"all of the training in the world\". What I am to point out that evenn if we were to brainwash her she would either still become currupt because that is whjat happens to ALL humans beings when they are given power over all. He then ends with a sad metaphorical statment that case at all.Defense 2My opponent then brings up Mitt Romney as an example, which does not help him. For one, abortion bans are a violation of women's civil right and another thing he brings up the risk that I stated in mthe last round. I said if Palin were to become a mindless drone of someone else other then herself then she would be under the control of one of three groups, the church and the corporations (in this case it could be both), which is what she already does if you take a look at her policies. Or the people themselves, in which there would be no reason to have a \"Queen of the Earth\" anyways. So really, my opponent has done nothing here but help me.Refutations1. My opponent has no point here, so I will dismiss it until he actually comes up with something that helps him.2. Puppies are irrelevant to this debate.
# Relevant Query: """
# FEWSHOT_3A_TOUCHE = "What are the arguments against concentrating global power in a single leader, using Sarah Palin as a hypothetical example?"

FEWSHOT_3H_SCIFACT = """Example 3:
Document: Reproducibility and automatic measurement of QT dispersion. This study investigated interobserver (two observers) and intrasubject (two measurements) reproducibility of QT dispersion from abnormal electrocardiograms in patients with previous myocardial infarction, and compared a user-interactive with an automatic measurement system. Standard 12-lead electrocardiograms, recorded at 25 mm.s-1, were randomly chosen from 70 patients following myocardial infarction. These were scanned into a personal computer, and specially designed software skeletonized and joined each image. The images were then available for user-interactive (mouse and computer screen), or automatic measurements using a specially designed algorithm. For all methods reproducibility of the RR interval was excellent (mean absolute errors 3-4 ms, relative errors 0.3-0.5%). Reproducibility of the mean QT interval was good; intrasubject error was 6 ms (relative error 1.4%), interobserver error was 7 ms (1.8%), and observers' vs automatic measurement errors were 10 and 11 ms (2.5, 2.8%). However QTc dispersion measurements had large errors for all methods; intrasubject error was 12 ms (17.3%), interobserver error was 15 ms (22.1%), and observers' vs automatic measurement were errors 30 and 28 ms (35.4, 31.9%). QT dispersion measurements rely on the most difficult to measure QT intervals, resulting in a problem of reproducibility. Any automatic system must not only recognize common T wave morphologies, but also these more difficult T waves, if it is to be useful for measuring QT dispersion. The poor reproducibility of QT dispersion limits its role as a useful clinical tool, particularly as a predictor of events.
Claim Supported By Document: """
FEWSHOT_3H_SENT_SCIFACT = """Example 3:
Document: Reproducibility and automatic measurement of QT dispersion. This study investigated interobserver (two observers) and intrasubject (two measurements) reproducibility of QT dispersion from abnormal electrocardiograms in patients with previous myocardial infarction, and compared a user-interactive with an automatic measurement system. Standard 12-lead electrocardiograms, recorded at 25 mm.s-1, were randomly chosen from 70 patients following myocardial infarction. These were scanned into a personal computer, and specially designed software skeletonized and joined each image. The images were then available for user-interactive (mouse and computer screen), or automatic measurements using a specially designed algorithm. For all methods reproducibility of the RR interval was excellent (mean absolute errors 3-4 ms, relative errors 0.3-0.5%). Reproducibility of the mean QT interval was good; intrasubject error was 6 ms (relative error 1.4%), interobserver error was 7 ms (1.8%), and observers' vs automatic measurement errors were 10 and 11 ms (2.5, 2.8%). However QTc dispersion measurements had large errors for all methods; intrasubject error was 12 ms (17.3%), interobserver error was 15 ms (22.1%), and observers' vs automatic measurement were errors 30 and 28 ms (35.4, 31.9%). QT dispersion measurements rely on the most difficult to measure QT intervals, resulting in a problem of reproducibility. Any automatic system must not only recognize common T wave morphologies, but also these more difficult T waves, if it is to be useful for measuring QT dispersion. The poor reproducibility of QT dispersion limits its role as a useful clinical tool, particularly as a predictor of events.
Sentence: The poor reproducibility of QT dispersion limits its role as a useful clinical tool, particularly as a predictor of events.
Claim Supported By Document: """
FEWSHOT_3A_SCIFACT = "Poor reproducibility of QT dispersion measurements limits its reliability as a clinical tool for predicting cardiac events, highlighting the need for improved measurement methods, especially for complex T wave morphologies."

# FEWSHOT_3H_NFCORPUS = """Example 3:
# Document: Milk consumption and acne in adolescent girls. There has been a remarkable paucity of evidence for an association between diet and acne. Our previous studies suggest that there is an association between milk intake and teenage acne. This is a prospective cohort study to evaluate that relationship. We studied 6,094 girls, aged 9-15 years in 1996, who reported dietary intake on up to three food frequency questionnaires from 1996 to 1998. Presence and severity of acne was assessed by questionnaire in 1999. We computed multivariate prevalence ratios (PR) and 95 percent confidence intervals for acne. After accounting for age at baseline, height and energy intake, the multivariate PRs (95 % CI; p-value for test of trend) for acne comparing highest (2 or more servings per day) to lowest (<1 per week) intake categories in 1996, were 1.20 (1.09, 1.31; <0.001) for total milk, 1.19 (1.06, 1.32; <0.001) for whole milk, 1.17 (1.04, 1.31; 0.002) for low fat milk and 1.19 (1.08, 1.31; <0.001) for skim milk. This result did not change appreciably when we excluded girls who reported use of contraceptives and when we restricted our analysis to those younger than 11 years of age at baseline. We found a positive association between intake of milk and acne. This finding supports earlier studies and suggests that the metabolic effects of milk are sufficient to elicit biological responses in consumers.
# Scientific Query: """
# FEWSHOT_3A_NFCORPUS = "Is there a positive association between milk consumption and acne prevalence in adolescent girls, and does this vary by type of milk (whole, low-fat, or skim)?"

FEWSHOT_3H_ARGUANA = """Example 3:
Document: Home-schooling is not the best option for exceptional students. The state does not ignore or abandon individuals that have special needs and those with special needs are those that most need the state's enormous resources to focus on their requirements. Once a student has needs of such a magnitude that demands it, they are educated in special schools specifically intended to help them, with staff trained to possess skills beyond that of a parent's instinct. Even if it were the case that home-schooling is better for the specific needs of exceptional students, the benefits of education in a wider context override the objection to class-based education. The experience of growing up alongside less and more able students produces individuals with greater understanding of their society1. 1'Teacher perceptions of mainstreaming/inclusion, 1958-1995: a research synthesis' Scruggs, Thomas E. Mastropieri, Margo A. Exceptional Children (1996)
Counter Argument:
"""
FEWSHOT_3H_SENT_ARGUANA = """Example 3:
Document: Home-schooling is not the best option for exceptional students. The state does not ignore or abandon individuals that have special needs and those with special needs are those that most need the state's enormous resources to focus on their requirements. Once a student has needs of such a magnitude that demands it, they are educated in special schools specifically intended to help them, with staff trained to possess skills beyond that of a parent's instinct. Even if it were the case that home-schooling is better for the specific needs of exceptional students, the benefits of education in a wider context override the objection to class-based education. The experience of growing up alongside less and more able students produces individuals with greater understanding of their society1. 1'Teacher perceptions of mainstreaming/inclusion, 1958-1995: a research synthesis' Scruggs, Thomas E. Mastropieri, Margo A. Exceptional Children (1996)
Sentence: The state does not ignore or abandon individuals that have special needs and those with special needs are those that most need the state's enormous resources to focus on their requirements.
Counter Argument:
"""
FEWSHOT_3A_ARGUANA = "Home-schooling is the best option for exceptional students, especially those with special needs, because it allows for a personalized, supportive environment that can be tailored to each child’s unique requirements, something that may be difficult to achieve in traditional school settings."
FEWSHOT_3H_ROBUST04 = """Example 3:
Document: BFN\n[Unattributed report: \"A Loose EU Is Not Necessarily to Our\nAdvantage\"]\n[Text] As a member of the European Union [EU], Finland\nmust not become subservient to the interests of any of the big\npowers in the EU. Finland must not align itself with either the\nBritish or the French ideology, but act as defender of the\ninterests of Europe's northeast corner. A loose EU is not\nnecessarily advantageous to Finland.\nThese were the points that SDP [Social Democratic Party]\nChairman Paavo Lipponen stressed in his speech at a seminar\norganized by the Trade and Industry Delegation on Wednesday [4\nMay]. Lipponen pointed out that Finland has eight months to\nprepare its membership policy and get ready to take full\nadvantage of membership. He added that Finland cannot function\nfor a single day in the EU without a clearly defined platform of\npolicy.\n\"Will it be the French or the British philosophy? The French\nfocus on finality, a clearly defined goal for a closely\nintegrated, federalist EU. The British prefer to advance\npragmatically and favor a loose community until this is proven\nwrong.\n\"Finland has the same right as the present member countries\nto hold on to its national interests. Good relations with\nRussia are too important an asset to Finland to be given up.\nThe EU cannot expect Finland to take on greater risks than any\nother member country. Our position has become easier due to the\nfact that the integration of Russia into European cooperation is\nalso vital to the EU.\"\nThe Foreign Minister Is Giving the Wrong Signals\nThe SPD chairman was critical about the attitude taken by\nForeign Minister Heikki Haavisto: \"The foreign minister has\nalready gone against Finland's interests on two occasions, once\nby hastening to declare himself a supporter of the British EU\nphilosophy and a second time by siding with the British view on\nthe minority vote issue.\"\nLipponen did not think it was self-evident at all that a\nloose community would be to Finland's advantage. This\nstandpoint is based on misconceptions from the Cold War, not on\na proper analysis of what is best for Finland: \"Finland would\nweaken its own influence by adopting a platform of opposition to\nfederalism. A Community without a common will and capacity to\nact is likely sooner or later to leave its smallest members in\nthe weakest position.\n\"In a loose EU, the great powers would skim the cream off\nthe\nmilk and focus on their own interests. If the new countries\nadopt an antifederal platform, it will strengthen the position\nof those who want to create an inner circle of `original' member\nstates and do business by means of coalitions.\"\nA loose EU, according to Lipponen, may even incite Germany\nto\ngo in its own direction.\nThe EU Must Expand to the East\n\"Germany is very much in favor of expanding the EU to the\nnorth and the east. This commits Finland, as a member of the\nEU, to take an early stance on the issue of the EU's expansion.\nThe newest members are generally opposed to the next\ncandidates. But here, Finland's interests are the same as those\nof the EU in general. The countries of East Europe must be\ntaken on board as members by the turn of the century. East\nEurope, which is heading for chaos as a result of insecurity and\npolitical and commercial discrimination, could become an\noverpowering threat to the EU and cause the disintegration of\nthe whole union.\"\nLipponen thinks that as a member of the EU, Finland must\nalso\nact as a staunch advocate of the interests of business and\nindustry. The most typical requirement in this respect concerns\nheavy industry, i.e., the paper, shipbuilding, and metal sectors.\nIn discussing Finland's role as defender of the interests of\nEurope's northeastern region, Lipponen referred to the EU's\npreparatory work to develop a basic structural plan for Europe:\n\"It does not include a single project addressing the attachment\nof the Baltic area to the Europe of the EU. The\nFinland-St.Petersburg-Moscow rail and road links, the\ndevelopment of the ports in the Gulf of Finland, and the Baltic\nroad network must be included in the Commission's project\nprogram.\"\n\"We Do Not Want a Eurohell\"\nLipponen stressed that Europe must be developed as a social\nproject, not only as a design engineered by a\npolitico-industrial elite: \"We do not want a Eurohell with a\ndogmatic economic and monetary policy, set to destroy the unity\nof the Community and where a political elite hands out\nmafia-style subsidies to mitigate opposition,\" he said.\nForeign Minister Heikki Haavisto believes that EU membership\nwill enable Finland to focus even more efficiently on the\ncentral goal of greater stability and security in Northern\nEurope. Haavisto, who also addressed the EVA [Business and\nIndustry Delegation] seminar, stressed the importance of\nFinland's role in influencing the way the EU develops its\nrelations with Russia and the Baltic states and in the design of\nEU's new northern dimension.\nHaavisto agreed with SPD Chairman Paavo Lipponen that\nFinland\nmust base its actions in the EU on the needs and interests of\nthe Finnish community: \"Only the Finns themselves will take\ncare of Finnish issues in Brussels,\" Haavisto added.\nHaavisto was definitely more cautious than Lipponen in\noutlining Finland's role in the EU. While Lipponen believes\nthat expansion to include East Europe would be in Finland's\ninterest, Haavisto confined himself to say that expansion will\nhave to be examined with care as it would affect our position in\nthe EU's northeast corner.\nHaavisto also dodged the issue of whether a loose or a close\nEU would be in Finland's interest. The question must be\nstudied, he said, alluding to the smaller EU states, of which\nmany have favored federative forms of decisionmaking on the\nbasis that this has committed the bigger countries to a closer\ncooperation.", "title": "Foreign Minister, SDP Chairman on EU Membership
Relavant Query: """
FEWSHOT_3H_SENT_ROBUST04 = """Example 3:
Document: BFN\n[Unattributed report: \"A Loose EU Is Not Necessarily to Our\nAdvantage\"]\n[Text] As a member of the European Union [EU], Finland\nmust not become subservient to the interests of any of the big\npowers in the EU. Finland must not align itself with either the\nBritish or the French ideology, but act as defender of the\ninterests of Europe's northeast corner. A loose EU is not\nnecessarily advantageous to Finland.\nThese were the points that SDP [Social Democratic Party]\nChairman Paavo Lipponen stressed in his speech at a seminar\norganized by the Trade and Industry Delegation on Wednesday [4\nMay]. Lipponen pointed out that Finland has eight months to\nprepare its membership policy and get ready to take full\nadvantage of membership. He added that Finland cannot function\nfor a single day in the EU without a clearly defined platform of\npolicy.\n\"Will it be the French or the British philosophy? The French\nfocus on finality, a clearly defined goal for a closely\nintegrated, federalist EU. The British prefer to advance\npragmatically and favor a loose community until this is proven\nwrong.\n\"Finland has the same right as the present member countries\nto hold on to its national interests. Good relations with\nRussia are too important an asset to Finland to be given up.\nThe EU cannot expect Finland to take on greater risks than any\nother member country. Our position has become easier due to the\nfact that the integration of Russia into European cooperation is\nalso vital to the EU.\"\nThe Foreign Minister Is Giving the Wrong Signals\nThe SPD chairman was critical about the attitude taken by\nForeign Minister Heikki Haavisto: \"The foreign minister has\nalready gone against Finland's interests on two occasions, once\nby hastening to declare himself a supporter of the British EU\nphilosophy and a second time by siding with the British view on\nthe minority vote issue.\"\nLipponen did not think it was self-evident at all that a\nloose community would be to Finland's advantage. This\nstandpoint is based on misconceptions from the Cold War, not on\na proper analysis of what is best for Finland: \"Finland would\nweaken its own influence by adopting a platform of opposition to\nfederalism. A Community without a common will and capacity to\nact is likely sooner or later to leave its smallest members in\nthe weakest position.\n\"In a loose EU, the great powers would skim the cream off\nthe\nmilk and focus on their own interests. If the new countries\nadopt an antifederal platform, it will strengthen the position\nof those who want to create an inner circle of `original' member\nstates and do business by means of coalitions.\"\nA loose EU, according to Lipponen, may even incite Germany\nto\ngo in its own direction.\nThe EU Must Expand to the East\n\"Germany is very much in favor of expanding the EU to the\nnorth and the east. This commits Finland, as a member of the\nEU, to take an early stance on the issue of the EU's expansion.\nThe newest members are generally opposed to the next\ncandidates. But here, Finland's interests are the same as those\nof the EU in general. The countries of East Europe must be\ntaken on board as members by the turn of the century. East\nEurope, which is heading for chaos as a result of insecurity and\npolitical and commercial discrimination, could become an\noverpowering threat to the EU and cause the disintegration of\nthe whole union.\"\nLipponen thinks that as a member of the EU, Finland must\nalso\nact as a staunch advocate of the interests of business and\nindustry. The most typical requirement in this respect concerns\nheavy industry, i.e., the paper, shipbuilding, and metal sectors.\nIn discussing Finland's role as defender of the interests of\nEurope's northeastern region, Lipponen referred to the EU's\npreparatory work to develop a basic structural plan for Europe:\n\"It does not include a single project addressing the attachment\nof the Baltic area to the Europe of the EU. The\nFinland-St.Petersburg-Moscow rail and road links, the\ndevelopment of the ports in the Gulf of Finland, and the Baltic\nroad network must be included in the Commission's project\nprogram.\"\n\"We Do Not Want a Eurohell\"\nLipponen stressed that Europe must be developed as a social\nproject, not only as a design engineered by a\npolitico-industrial elite: \"We do not want a Eurohell with a\ndogmatic economic and monetary policy, set to destroy the unity\nof the Community and where a political elite hands out\nmafia-style subsidies to mitigate opposition,\" he said.\nForeign Minister Heikki Haavisto believes that EU membership\nwill enable Finland to focus even more efficiently on the\ncentral goal of greater stability and security in Northern\nEurope. Haavisto, who also addressed the EVA [Business and\nIndustry Delegation] seminar, stressed the importance of\nFinland's role in influencing the way the EU develops its\nrelations with Russia and the Baltic states and in the design of\nEU's new northern dimension.\nHaavisto agreed with SPD Chairman Paavo Lipponen that\nFinland\nmust base its actions in the EU on the needs and interests of\nthe Finnish community: \"Only the Finns themselves will take\ncare of Finnish issues in Brussels,\" Haavisto added.\nHaavisto was definitely more cautious than Lipponen in\noutlining Finland's role in the EU. While Lipponen believes\nthat expansion to include East Europe would be in Finland's\ninterest, Haavisto confined himself to say that expansion will\nhave to be examined with care as it would affect our position in\nthe EU's northeast corner.\nHaavisto also dodged the issue of whether a loose or a close\nEU would be in Finland's interest. The question must be\nstudied, he said, alluding to the smaller EU states, of which\nmany have favored federative forms of decisionmaking on the\nbasis that this has committed the bigger countries to a closer\ncooperation.", "title": "Foreign Minister, SDP Chairman on EU Membership
Sentence: Finland must not align itself with either the British or the French ideology, but act as defender of the interests of Europe's northeast corner. A loose EU is not necessarily advantageous to Finland.
Relavant Query: """
FEWSHOT_3A_ROBUST04 = "What stance does Finland's SDP Chairman Paavo Lipponen take on a loose versus closely integrated EU, and how does this relate to Finland's interests?"

# FEWSHOT_3H_SCIDOCS = """Example 3:
# Document: Review of Inflatable Booms for Deployable Space Structures : Packing and Rigidization Inflatable structures offer the potential of compactly stowing lightweight structures, which assume a fully deployed state in space. An important category of space inflatables are cylindrical booms, which may form the structural members of trusses or the support structure for solar sails. Two critical and interdependent aspects of designing inflatable cylindrical booms for space applications are i) packaging methods that enable compact stowage and ensure reliable deployment, and ii) rigidization techniques that provide long-term structural ridigity after deployment. The vast literature in these two fields is summarized to establish the state of the art.
# Title of the paper citing the document: """
# FEWSHOT_3A_SCIDOCS = "One-DOF Superimposed Rigid Origami with Multiple States"
# FEWSHOT_3H_SCIDOCS = """Example 3:
# Document: A Lightweight Semantic Web-based Approach for Data Annotation on IoT Gateways Internet of Things (IoT) applications rely on networks composed of set of heterogeneous sensors and smart devices, which have the capability to constantly, observe the surroundings and gather data. This heterogeneity is reflected on raw data gathered by such type of systems. Consequently, the task of high-level IoT applications to interpret such data and detect events in the real world is more complex. Moreover, data heterogeneity leads to the lack of interoperability between IoT applications. Semantic Web (SW) technologies have been widely adopted to model and integrate data from different sources on the web; extending them to the IoT domain can be used to mitigate the aforementioned challenges. Semantically annotating IoT data is a fundamental step toward developing smarter and interoperable IoT applications. However, this type of process requires a large amount of computing resources, especially in scenarios where a large number of sensors is expected to be involved such as smart city. To address these challenges, we propose a lightweight semantic annotation approach that can be implemented on resource-constrained IoT gateways connected to a limited number of sensors. To evaluate the feasibility of the proposed approach, we have carried out a set of experimentations using a middleware prototype implementation. Several benchmarks are considered such as: Data size, response time, and resource utilization. c \u00a9 2017 The Authors. Published by Elsevier B.V. .
# Title of the paper citing the document: """
# FEWSHOT_3A_SCIDOCS = "Scalable Semantic Annotation for High-Density IoT Networks: A Lightweight Approach for Enhanced Interoperability in Smart City Applications"
FEWSHOT_3H_SCIDOCS = """Example 3:
Document: Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning We present Deep Voice 3, a fully-convolutional attention-based neural textto-speech (TTS) system. Deep Voice 3 matches state-of-the-art neural speech synthesis systems in naturalness while training an order of magnitude faster. We scale Deep Voice 3 to dataset sizes unprecedented for TTS, training on more than eight hundred hours of audio from over two thousand speakers. In addition, we identify common error modes of attention-based speech synthesis networks, demonstrate how to mitigate them, and compare several different waveform synthesis methods. We also describe how to scale inference to ten million queries per day on a single GPU server.
Relevant Query: """
FEWSHOT_3H_SENT_SCIDOCS = """Example 3:
Document: Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning We present Deep Voice 3, a fully-convolutional attention-based neural textto-speech (TTS) system. Deep Voice 3 matches state-of-the-art neural speech synthesis systems in naturalness while training an order of magnitude faster. We scale Deep Voice 3 to dataset sizes unprecedented for TTS, training on more than eight hundred hours of audio from over two thousand speakers. In addition, we identify common error modes of attention-based speech synthesis networks, demonstrate how to mitigate them, and compare several different waveform synthesis methods. We also describe how to scale inference to ten million queries per day on a single GPU server.
Sentence: We also describe how to scale inference to ten million queries per day on a single GPU server.
Relevant Query: """
FEWSHOT_3A_SCIDOCS = "Text-to-Speech System for Visually Impaired People: A Survey"

FEWSHOT_3H_NQ = """Example 3:
Document: List of High Courts of India The Calcutta High Court is the oldest High Court in the country, established on 2 July 1862. High Courts that handle a large number of cases of a particular region have permanent benches established there. Benches are also present in states which come under the jurisdiction of a court outside its territorial limits. Smaller states with few cases may have circuit benches established. Circuit benches (known as circuit courts in some parts of the world) are temporary courts which hold proceedings for a few selected months in a year. Thus cases built up during this interim period are judged when the circuit court is in session. According to a study conducted by Bengaluru-based NGO Daksh on 21 high courts in collaboration with the Ministry of Law and Justice in March 2015, it was found that average pendency of a case in High courts in India is 3 years.[2]
Relevant Query: """
FEWSHOT_3H_SENT_NQ = """Example 3:
Document: List of High Courts of India The Calcutta High Court is the oldest High Court in the country, established on 2 July 1862. High Courts that handle a large number of cases of a particular region have permanent benches established there. Benches are also present in states which come under the jurisdiction of a court outside its territorial limits. Smaller states with few cases may have circuit benches established. Circuit benches (known as circuit courts in some parts of the world) are temporary courts which hold proceedings for a few selected months in a year. Thus cases built up during this interim period are judged when the circuit court is in session. According to a study conducted by Bengaluru-based NGO Daksh on 21 high courts in collaboration with the Ministry of Law and Justice in March 2015, it was found that average pendency of a case in High courts in India is 3 years.[2]
Sentence: The Calcutta High Court is the oldest High Court in the country, established on 2 July 1862.
Relevant Query: """
FEWSHOT_3A_NQ = "What is the oldest High Court in India, and how do High Courts manage cases in regions with lower caseloads?"

# FEWSHOT_3H_NQ = """Example 3:
# Document: """

FEWSHOT_3H_QUORA = """Example 3:
Question: When Obama leaves office, will he give up the @POTUS account on Twitter?
Duplicate Question: """
FEWSHOT_3A_QUORA = (
    "After Obama’s term ends, will he hand over the @POTUS Twitter handle?"
)

FEWSHOT_3H_NEWS = """Example 2:
Document: Title: Fiorina dismisses Planned Parenthood criticism, knocks Clinton, Obama and congressional Republicans Content: GREENVILLE, S.C. \u2014 Republican presidential candidate Carly Fiorina forcefully dismissed charges that the controversial Planned Parenthood video she described during Wednesday\u2019s GOP primary debate does not exist, earning several standing ovations at a presidential forum here in Greenville as she tore into President Obama and Democratic candidate Hillary Rodham Clinton for supporting the organization. \u201cYes ladies and gentlemen, they are real, and I will issue my charge again: Hillary Clinton, Barack Obama, anyone who wants to defend Planned Parenthood, watch these tapes,\u201d Fiorina told an audience at the Heritage Action \"Take Back America\" event, which replied with enormous applause. \u201cIt is not actually about being pro-choice or pro-life. We cannot be a nation that funds this kind of barbarity, and that is what it is.\u201d During Wednesday's debate, Fiorina vividly described watching a videotape that showed \"a fully formed fetus, it's heart beating, it's legs kicking while someone says we have to keep it alive to harvest its brain.\" It was a powerful moment for Fiorina but, according to a review of the existing videos by The Washington Post and other media outlets, no video exists showing such a procedure. *[WSU]: Wayne State University *[Tue]: Tuesday *[Wed]: Wednesday *[UNM]: University of New Mexico The former Hewlett Packard CEO also had strong words for congressional Republicans, whom she urged to fight to defund Planned Parenthood amid ongoing budget negotiations. Such a standoff would potentially lead to a government shutdown. \u201cWe\u2019re in charge of the Senate and we have record majorities in the House. That\u2019s a huge change from 2013. And by the way, I worked really hard for that change and I know all of you really worked hard for that change too,\u201d she said. \u201cAnd what\u2019s disappointing is we\u2019re having the same old conversation. Nothing has changed.\u201d Current U.S. law already bans the use of any federal funds for abortions, which constitute a small portion of Planned Parenthood services. Republicans are now pushing to ban all finding of the organization, which provides a wide range of women's health services. With an eye toward a potential general election match-up against Clinton, who is perhaps her favorite target on the stump, Fiorina also accused the former secretary of state of hypocrisy on women\u2019s rights issues. \u201cHow hypocritical of Hillary Clinton, who is running on her record of protecting women\u2019s rights when in her position as secretary of state she promptly took off the table women\u2019s rights, human rights, and any other topic of conversation that might be uncomfortable for the Chinese,\u201d Fiorina said. \u201cRest assured, ladies and gentlemen, I will bring that up in a general election debate stage.\u201d
Relevant Article Headline:
"""
FEWSHOT_3A_NEWS = """Carly Fiorina defends disputed Planned Parenthood video, urges Republicans to defund the group and criticizes Clinton and Obama"""

FEWSHOT_3H_DB = """Example 3:
Document: Ole Herman Johannes Krag Ole Herman Johannes Krag (7 April 1837 -  9 December 1916) was a Norwegian officer and firearms designer.
Relevant Query:
"""
FEWSHOT_3A_DB = """Ole Herman Johannes Krag"""

FEWSHOT_3H_HOTPOT = """Example 3:
Document: Miklos Porkolab Miklos Porkolab (born March 24, 1939) is a Hungarian-American physicist specializing in plasma physics. He emigrated in 1957 from Hungary to Canada, where he studied at the University of British Columbia (Bachelor, 1963) and then at Stanford University, where he obtained his Master degree in 1964 and his PhD in 1968. He then moved to the Princeton Plasma Physics Laboratory, where he worked as a Senior Research Physicist until 1975. During the following year, Porkolab worked at the Max Planck Institute for Plasma Physics in Garching, Germany, under the auspices of the Humboldt Foundation as a winner of the \"US Senior Scientist Award\". In 1977 he became Professor of Physics at the Massachusetts Institute of Technology, where he later led the Plasma Science and Fusion Center (PSFC) for many years.
Relevant Query:
"""
FEWSHOT_3A_HOTPOT = "Which center at MIT did Miklos Porkolab later lead for many years?"

FEWSHOT_3H_MAP = {
    "fiqa": FEWSHOT_3H,
    "trec-covid": FEWSHOT_3H,
    "webis-touche2020": FEWSHOT_3H,
    "scifact": FEWSHOT_3H_SCIFACT,
    "nfcorpus": FEWSHOT_3H,
    "robust04": FEWSHOT_3H_ROBUST04,
    "arguana": FEWSHOT_3H_ARGUANA,
    "scidocs": FEWSHOT_3H_SCIDOCS,
    "nq": FEWSHOT_3H_NQ,
    "quora": FEWSHOT_3H_QUORA,
    "trec-news": FEWSHOT_3H_NEWS,
    "cqadupstack-android": FEWSHOT_3H,
    "cqadupstack-english": FEWSHOT_3H,
    "cqadupstack-gaming": FEWSHOT_3H,
    "cqadupstack-gis": FEWSHOT_3H,
    "cqadupstack-mathematica": FEWSHOT_3H,
    "cqadupstack-physics": FEWSHOT_3H,
    "cqadupstack-programmers": FEWSHOT_3H,
    "cqadupstack-stats": FEWSHOT_3H,
    "cqadupstack-tex": FEWSHOT_3H,
    "cqadupstack-unix": FEWSHOT_3H,
    "cqadupstack-webmasters": FEWSHOT_3H,
    "cqadupstack-wordpress": FEWSHOT_3H,
    "dbpedia-entity": FEWSHOT_3H_DB,
    "hotpotqa": FEWSHOT_3H_HOTPOT,
}
FEWSHOT_3H_SENT_MAP = {
    "fiqa": FEWSHOT_3H_SENT,
    "trec-covid": FEWSHOT_3H_SENT,
    "webis-touche2020": FEWSHOT_3H_SENT,
    "scifact": FEWSHOT_3H_SENT_SCIFACT,
    "nfcorpus": FEWSHOT_3H_SENT,
    "robust04": FEWSHOT_3H_SENT_ROBUST04,
    "arguana": FEWSHOT_3H_SENT_ARGUANA,
    "scidocs": FEWSHOT_3H_SENT_SCIDOCS,
    "nq": FEWSHOT_3H_SENT_NQ,
    "quora": FEWSHOT_3H_QUORA,
}
FEWSHOT_3A_MAP = {
    "fiqa": FEWSHOT_3A,
    "trec-covid": FEWSHOT_3A,
    "webis-touche2020": FEWSHOT_3A,
    "scifact": FEWSHOT_3A_SCIFACT,
    "nfcorpus": FEWSHOT_3A,
    "robust04": FEWSHOT_3A_ROBUST04,
    "arguana": FEWSHOT_3A_ARGUANA,
    "scidocs": FEWSHOT_3A_SCIDOCS,
    "nq": FEWSHOT_3A_NQ,
    "quora": FEWSHOT_3A_QUORA,
    "trec-news": FEWSHOT_3A_NEWS,
    "cqadupstack-android": FEWSHOT_3A,
    "cqadupstack-english": FEWSHOT_3A,
    "cqadupstack-gaming": FEWSHOT_3A,
    "cqadupstack-gis": FEWSHOT_3A,
    "cqadupstack-mathematica": FEWSHOT_3A,
    "cqadupstack-physics": FEWSHOT_3A,
    "cqadupstack-programmers": FEWSHOT_3A,
    "cqadupstack-stats": FEWSHOT_3A,
    "cqadupstack-tex": FEWSHOT_3A,
    "cqadupstack-unix": FEWSHOT_3A,
    "cqadupstack-webmasters": FEWSHOT_3A,
    "cqadupstack-wordpress": FEWSHOT_3A,
    "dbpedia-entity": FEWSHOT_3A_DB,
    "hotpotqa": FEWSHOT_3A_HOTPOT,
}

FEWSHOT_4H = """Example 4:
Document: {document}
Relevant Query: """
FEWSHOT_4H_KW = """Example 4:
Document: {document}
Keywords: {keywords}
Relevant Query: """
FEWSHOT_4H_SENT = """Example 4:
Document: {document}
Sentence: {sentence}
Relevant Query: """
FEWSHOT_4H_KWGEN = """Example 4:
Document: {document}
Keywords: 1."""
FEWSHOT_4H_SM = """Example 4:
Document: {document}
Summary: {summary}
Relevant Query: """
FEWSHOT_4H_SUMMARY = """Example 4:
Summary: {summary}
Relevant Query: """
FEWSHOT_4H_STREAMLINED = """Example 4:
Clarified Document: {document}
Relevant Query: """
FEWSHOT_4H_SCIFACT = """Example 4:
Document: {document}
Claim Supported By Document: """
FEWSHOT_4H_SENT_SCIFACT = """Example 4:
Document: {document}
Sentence: {sentence}
Claim Supported By Document: """
# FEWSHOT_4H_NFCORPUS = """Example 4:
# Document: {document}
# Scientific Query: """
FEWSHOT_4H_ARGUANA = """Example 4:
Document: {document}
Counter Argument: """
FEWSHOT_4H_SENT_ARGUANA = """Example 4:
Document: {document}
Sentence: {sentence}
Counter Argument: """
# FEWSHOT_4H_SCIDOCS = """Example 4:
# Document: {document}
# Title of the paper citing the document: """
FEWSHOT_4H_SCIDOCS = """Example 4:
Document: {document}
Relevant Query: """
FEWSHOT_4H_QUORA = """Example 4:
Question: {document}
Duplicate Question: """
FEWSHOT_4H_SENT_QUORA = """Example 4:
Question: {document}
Duplicate Question: """
FEWSHOT_4H_NEWS = """Example 4:
Document: {document}
Relevant Article Headline: """
FEWSHOT_4H_DB = """Example 4:
Document: {document}
Relevant Query: """
FEWSHOT_4H_MAP = {
    "fiqa": FEWSHOT_4H,
    "trec-covid": FEWSHOT_4H,
    "webis-touche2020": FEWSHOT_4H,
    "scifact": FEWSHOT_4H_SCIFACT,
    "nfcorpus": FEWSHOT_4H,
    "robust04": FEWSHOT_4H,
    "arguana": FEWSHOT_4H_ARGUANA,
    "scidocs": FEWSHOT_4H_SCIDOCS,
    "nq": FEWSHOT_4H,
    "quora": FEWSHOT_4H_QUORA,
    "trec-news": FEWSHOT_4H_NEWS,
    "cqadupstack-android": FEWSHOT_4H,
    "cqadupstack-english": FEWSHOT_4H,
    "cqadupstack-gaming": FEWSHOT_4H,
    "cqadupstack-gis": FEWSHOT_4H,
    "cqadupstack-mathematica": FEWSHOT_4H,
    "cqadupstack-physics": FEWSHOT_4H,
    "cqadupstack-programmers": FEWSHOT_4H,
    "cqadupstack-stats": FEWSHOT_4H,
    "cqadupstack-tex": FEWSHOT_4H,
    "cqadupstack-unix": FEWSHOT_4H,
    "cqadupstack-webmasters": FEWSHOT_4H,
    "cqadupstack-wordpress": FEWSHOT_4H,
    "dbpedia-entity": FEWSHOT_4H_DB,
    "hotpotqa": FEWSHOT_4H,
}
FEWSHOT_4H_SENT_MAP = {
    "fiqa": FEWSHOT_4H_SENT,
    "trec-covid": FEWSHOT_4H_SENT,
    "webis-touche2020": FEWSHOT_4H_SENT,
    "scifact": FEWSHOT_4H_SENT_SCIFACT,
    "nfcorpus": FEWSHOT_4H_SENT,
    "robust04": FEWSHOT_4H_SENT,
    "arguana": FEWSHOT_4H_SENT_ARGUANA,
    "scidocs": FEWSHOT_4H_SENT,
    "nq": FEWSHOT_4H_SENT,
    "quora": FEWSHOT_4H_SENT_QUORA,
}


FEWSHOT_1H_SUMMARY_DECISION = """
Document: You\'re missing a very important thing: YEAR END values in (U.S.) $ millions unless otherwise noted So 7098 is not $7,098.  That would be a rather silly amount for Coca Cola to earn in a year don\'t you think?  I mean, some companies might happen upon random small income amounts, but it seems pretty reasonable to assume they\'ll earn (or lose) millions or billions, not thousands. This is a normal thing to do on reports like this; it\'s wasteful to calculate to so many significant digits, so they divide everything by 1000 or 1000000 and report at that level.  You need to look on the report (usually up top left, but it can vary) to see what factor they\'re dividing by. Coca Cola\'s earnings per share are $1.60 for FY 2014, which is 7,098/4450 (use the whole year numbers, not the quarter 4 numbers; and here they\'re both in millions, so they divide out evenly).   You also need to understand that ""Dividend on preferred stock"" is not the regular dividend; I don\'t see it explicitly called out on the page you reference. They may not have preferred stock and/or may not pay dividends on it in excess of common stock (or at all).
Reasoning: """
FEWSHOT_1A_SUMMARY_DECISION = """
This document primarily consists of an explanation regarding financial reporting, specifically how figures are presented in year-end reports, the typical omission of smaller units (such as thousands), and the need to understand certain terms like "Dividend on preferred stock." It delves into specific reporting conventions rather than presenting a direct, concise fact or assertion.

To generate queries effectively, summarization would help streamline the core ideas and remove extraneous explanations, yielding a more targeted query set that focuses on the key financial reporting conventions.

Decision: Yes"""
FEWSHOT_2H_SUMMARY_DECISION = """Example 2:
Document: All people in america are incurably stupid Its unbelievable how stupid people are today, especially in America.
Reasoning: """
FEWSHOT_2A_SUMMARY_DECISION = """
The document contains a broad, biased statement without providing any specific information, context, or detail that would make it valuable for generating informative or effective queries. The content lacks substantive information, rendering it ineffective for query generation focused on nuanced or factual inquiry.

Decision: No"""
SUMMARY_DECISION_INSTRUCT = 'Analyze the following document and decide whether it needs summarization for effective query generation. Provide your reasoning and you must end with "Yes" or "No"'
FEWSHOT_4H_SUMMARY_DECISION = """
Document: {document}
Decision: """


async def summary_decision(doc, client) -> bool:
    parser = SummaryDecisionOutputParser()
    prompt = FEWSHOT_4H_SUMMARY_DECISION.format(document=doc).rstrip()
    chat_prompt_with_inst = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=SUMMARY_DECISION_INSTRUCT),
            HumanMessage(content=FEWSHOT_1H_SUMMARY_DECISION),
            AIMessage(content=FEWSHOT_1A_SUMMARY_DECISION),
            HumanMessage(content=FEWSHOT_2H_SUMMARY_DECISION),
            AIMessage(content=FEWSHOT_2A_SUMMARY_DECISION),
            HumanMessage(content=prompt),
        ]
    )
    chain = chat_prompt_with_inst | client | parser
    chain = chain.with_retry()
    result = await chain.ainvoke({})
    # try:
    #     retryable_chain = RunnableRetry(
    #         bound=chain,
    #         retry_exception_types=(OutputParserException,),
    #         max_attempt_number=5,
    #         wait_exponential_jitter=True,
    #     )
    #     result = await retryable_chain.ainvoke({})
    # except OutputParserException as e:
    #     print(e)
    #     raise e
    return result


async def summary_gen_async(doc, client):
    chat_prompt_with_inst = ChatPromptTemplate.from_messages(
        [
            # SystemMessage(content=INSTRUCTION),
            # HumanMessage(content=FEWSHOT_HUMAN),
            # AIMessage(content=FEWSHOT_AI),
            HumanMessagePromptTemplate.from_template("{user_message}"),
        ]
    )
    user_prompt_pos = {
        "user_message": f"Document: {doc}\n\nBased on the document, write a summary:"
    }
    chain = chat_prompt_with_inst | client
    result = await chain.ainvoke(user_prompt_pos)
    content = (
        result.content.split("Here is a summary of the document:")[-1]
        .strip()
        .split("The summary is:")[-1]
        .strip()
        .split("Here is a summary based on the document:")[-1]
        .strip()
        .split("Here is a summary:")[-1]
        .strip()
    )
    return content


STREAMLINE_INSTRUCTION = """
Please rewrite the following document to make it easier to understand, summarizing only the most important information. If the document is already clear and concise, you may leave it unchanged.
Think you are the author of the document.
"""

# STREAMLINE_INSTRUCTION_TRECCOVID = """
# Please provide a concise summary of the following biomedical document, highlighting the main objectives, methods, results, and conclusions. Identify the key concepts and keywords.
# """

STREAMLINE_INSTRUCTION_TOUCHE = """
Improve the writing so that it is more easy to understand. Write it as if you are the author of the document.
"""
# STREAMLINE_INSTRUCTION_TOUCHE = """
# Rewrite the document about argument to make it easier to understand, concise and coherent. Write it as if you are the author of the document.
# """
# Please provide a concise summary of the following argumentative text, focusing on the key arguments or thesis, supporting arguments and evidence, conclusions drawn which identifies potential user information needs or questions that would lead someone to seek this document.

# STREAMLINE_INSTRUCTION_SCIFACT = """
# Identify the key facts, figures, and entities mentioned in the document, including important relationships and findings.
# """

# STREAMLINE_INSTRUCTION_NFCORPUS = """
# Please provide a concise summary of the following biomedical document, highlighting the main objectives, methods, results, and conclusions which identifies the key concepts and keywords.
# """

# STREAMLINE_INSTRUCTION_ARGUANA = """
# Please read the following document carefully and produce a lexical abstraction of it. The lexical abstraction should:

# Explain the core argument of the document in a clear and concise manner.
# Rewrite the content so that it is easier to follow and understand.
# Capture the main claim or thesis, along with the key supporting points and evidence.
# Simplify complex sentences and terminology without losing the original meaning.
# Maintain the logical flow of the original argument.
# Avoid unnecessary details or tangents, focusing on the essential points.
# Your goal is to make the core argument accessible and understandable to a broad audience while preserving the original intent and nuances.
# """
STREAMLINE_INSTRUCTION_ARGUANA = """
Write a concise and coherent summary of the original argument.
Use neutral and objective language to present the original viewpoint.
Avoid incorporating any biases or judgments from the counter-argument.
"""
STREAMLINE_INSTRUCTION_SCIDOCS = """
Write a concise and coherent summary of the research document.
summary should contain the main objectives, methods, results, and conclusions.
"""
# STREAMLINE_INSTRUCTION_SCIDOCS = """
# Please rewrite the research document to make it easier to understand. If the document is already clear and concise, you may leave it unchanged. Think you are the author of the document.
# """
# STREAMLINE_INSTRUCTION_SCIDOCS = """
# Please read the following document carefully and produce a lexical abstraction that explains the key contributions of the paper and suggests possible future works. The lexical abstraction should be a clear and concise summary that highlights the main findings and significance of the research, simplifies complex concepts and terminology to enhance readability, and maintains the logical flow and coherence of the original document. It should focus on essential points without unnecessary details, making the content accessible to a broad audience while preserving the original intent and nuances.
# """

STREAMLINE_INSTRUCTION_NQ = """
Write a concise and coherent summary of the wikipedia article."""

# summary should contain the main topics, concepts, and keywords.

STREAMLINE_INSTRUCTION_QUORA = """
Create multiple paraphrased versions of the given question that maintain the same meaning but use different wording and sentence structures."""

STREAMLINE_INSTRUCTION_MAP = {
    "fiqa": STREAMLINE_INSTRUCTION,
    "trec-covid": STREAMLINE_INSTRUCTION,
    "webis-touche2020": STREAMLINE_INSTRUCTION_TOUCHE,
    "scifact": STREAMLINE_INSTRUCTION,
    "nfcorpus": STREAMLINE_INSTRUCTION,
    "robust04": STREAMLINE_INSTRUCTION,
    "arguana": STREAMLINE_INSTRUCTION_ARGUANA,
    # "scidocs": STREAMLINE_INSTRUCTION,
    "scidocs": STREAMLINE_INSTRUCTION_SCIDOCS,
    "nq": STREAMLINE_INSTRUCTION_NQ,
    "quora": STREAMLINE_INSTRUCTION_QUORA,
}


async def streamline_summary_gen_async(doc, client):
    chat_prompt_with_inst = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=STREAMLINE_INSTRUCTION),
            HumanMessagePromptTemplate.from_template("{user_message}"),
        ]
    )
    user_prompt_pos = {"user_message": f"Document: {doc}\n\nOutput:"}
    chain = chat_prompt_with_inst | client
    result = await chain.ainvoke(user_prompt_pos)
    content = remove_intro_line(result.content)
    return content


async def streamline_summary_gen_async_dataset(doc, client, dataset):
    chat_prompt_with_inst = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=STREAMLINE_INSTRUCTION_MAP[dataset]),
            HumanMessagePromptTemplate.from_template("{user_message}"),
        ]
    )
    user_prompt_pos = {"user_message": f"Document: {doc}\n\nOutput:"}
    chain = chat_prompt_with_inst | client
    result = await chain.ainvoke(user_prompt_pos)
    content = remove_intro_line(result.content)
    return content


FEWSHOT_STRM_1H_SCIDOCS = """Example 1:
Document: Thermal Facial Analysis for Deception Detection Thermal imaging technology can be used to detect stress levels in humans based on the radiated heat from their face. In this paper, we use thermal imaging to monitor the periorbital region's thermal variations and test whether it can offer a discriminative signature for detecting deception. We start by presenting an overview on automated deception detection and propose a novel methodology, which we validate experimentally on 492 thermal responses (249 lies and 243 truths) extracted from 25 participants. The novelty of this paper lies in scoring a larger number of questions per subject, emphasizing a within-person approach for learning from data, proposing a framework for validating the decision making process, and correct evaluation of the generalization performance. A $k$ -nearest neighbor classifier was used to classify the thermal responses using different strategies for data representation. We report an 87% ability to predict the lie/truth responses based on a within-person methodology and fivefold cross validation. Our results also show that the between-person approach for modeling deception does not generalize very well across the training data.
Limitation of the research: """
FEWSHOT_STRM_1A_SCIDOCS = """The study's participant pool is relatively small, consisting of only 25 individuals, which limits the diversity and generalizability of the findings to broader populations. Additionally, the research highlights that the between-person approach for modeling deception does not generalize well, indicating limitations in applying the findings across diverse individuals. The controlled environment in which the experiments were likely conducted may not reflect real-world conditions where external factors, such as movement, lighting, or environmental stressors, could influence thermal readings.

The study focuses solely on thermal imaging and the periorbital region's thermal variations, which may restrict the robustness of the approach compared to multimodal methods that incorporate additional physiological or behavioral cues. Evaluating deception as a binary outcome of lie or truth oversimplifies the complex nature of human behavior and stress responses during deception. The high accuracy achieved using a within-person methodology may lead to overfitting, making it less effective for unseen individuals.

Furthermore, the study does not address ethical and privacy concerns, which are critical when applying thermal imaging for deception detection in real-world scenarios, such as security or law enforcement. Finally, the absence of longitudinal analysis leaves unanswered questions about the consistency of thermal responses over time or under varying conditions for the same individuals, which could affect the reliability of the methodology."""
# FEWSHOT_STRM_1H_SCIDOCS = """Example 1:
# Document: Thermal Facial Analysis for Deception Detection Thermal imaging technology can be used to detect stress levels in humans based on the radiated heat from their face. In this paper, we use thermal imaging to monitor the periorbital region's thermal variations and test whether it can offer a discriminative signature for detecting deception. We start by presenting an overview on automated deception detection and propose a novel methodology, which we validate experimentally on 492 thermal responses (249 lies and 243 truths) extracted from 25 participants. The novelty of this paper lies in scoring a larger number of questions per subject, emphasizing a within-person approach for learning from data, proposing a framework for validating the decision making process, and correct evaluation of the generalization performance. A $k$ -nearest neighbor classifier was used to classify the thermal responses using different strategies for data representation. We report an 87% ability to predict the lie/truth responses based on a within-person methodology and fivefold cross validation. Our results also show that the between-person approach for modeling deception does not generalize very well across the training data.
# Brainstorm the most plausible future research topic: """
# FEWSHOT_STRM_1A_SCIDOCS = """Building on the current findings, future research could focus on developing a real-time deception detection system that integrates thermal imaging with other physiological and behavioral modalities, such as voice stress analysis, facial micro-expression detection, or eye movement tracking. The goal would be to enhance accuracy and generalization across diverse populations by combining complementary data sources. Key directions could include:

# Investigating advanced machine learning techniques, such as deep learning models, to extract more nuanced patterns from thermal data.
# Exploring adaptive methods to handle individual differences in thermal and physiological responses, improving the generalizability of the system.
# Expanding the dataset to include a larger and more diverse participant pool to assess the robustness of the methodology across demographic and cultural variations.
# Developing frameworks for real-time data acquisition and decision-making, ensuring system usability in practical scenarios, such as law enforcement or airport security."""
FEWSHOT_STRM_1H_ARGUANA = """Example 1:
Document: ployment tax education university house would fund provision higher education The main problem with the proposition argument is the belief that a graduate will be earning \u00a340,000 immediately after leaving university, this is clearly not the case, particularly in the current economic climate, the average starting wage for a graduate was in 2009 \u00a323,500 with only one in ten exceeding \u00a336,000. (Milkround, 2009) The argument does in part accept this weakness however what it does not point out is that many careers which require a university degree may never pay greater than \u00a340,000. What a graduate tax focuses on is getting a job after university, this is not always the reason that people wish to go to university, take for example a mature student who just wants to self-better themselves, could they still get access to education when the system would be built upon getting young people into work? University should not be commoditized, it should be considered sacred in its own right; introducing a graduate tax turns university into a means to get a career rather than being a place of pure education.
Main Argument and Supporting Evidence: """
FEWSHOT_STRM_1A_ARGUANA = """"""
FEWSHOT_STRM_1H_QUORA = """Example 1:
Question: How do you know if you are spiritually enlightened?
Rewritten Questions: """
FEWSHOT_STRM_1A_QUORA = """How can you recognize signs of spiritual enlightenment in yourself?
What are the indicators that you may have reached spiritual enlightenment?
How do you determine if you have achieved spiritual awakening?
What does it feel like to experience spiritual enlightenment?
How can one identify personal growth toward spiritual enlightenment?
What are the characteristics of a spiritually enlightened state?
How do you know if you are progressing on the path to spiritual enlightenment?
What changes in mindset or behavior suggest spiritual enlightenment?
How can you discern if you have reached a higher state of spiritual awareness?
What internal or external signs point to spiritual enlightenment?"""
# FEWSHOT_STRM_1H_QUORA = """Example 1:
# Question: How do you know if you are spiritually enlightened?
# Rewritten Questions: """
# FEWSHOT_STRM_1A_QUORA = """How can you recognize signs of spiritual enlightenment in yourself?
# What are the indicators that you may have reached spiritual enlightenment?
# How do you determine if you have achieved spiritual awakening?
# What does it feel like to experience spiritual enlightenment?
# How can one identify personal growth toward spiritual enlightenment?
# What are the characteristics of a spiritually enlightened state?
# How do you know if you are progressing on the path to spiritual enlightenment?
# What changes in mindset or behavior suggest spiritual enlightenment?
# How can you discern if you have reached a higher state of spiritual awareness?
# What internal or external signs point to spiritual enlightenment?"""

FEWSHOT_STRM_1H_MAP = {
    "scidocs": FEWSHOT_STRM_1H_SCIDOCS,
    "arguana": FEWSHOT_STRM_1H_ARGUANA,
    "quora": FEWSHOT_STRM_1H_QUORA,
}
FEWSHOT_STRM_1A_MAP = {
    "scidocs": FEWSHOT_STRM_1A_SCIDOCS,
    "arguana": FEWSHOT_STRM_1A_ARGUANA,
    "quora": FEWSHOT_STRM_1A_QUORA,
}

FEWSHOT_STRM_2H_SCIDOCS = """Example 2:
Document: Vision-Based Gesture Recognition: A Review The use of gesture as a natural interface serves as a motivating force for research in modeling, analyzing and recognition of gestures. In particular, human computer intelligent interaction needs vision-based gesture recognition, which involves many interdisciplinary studies. A survey on recent vision-based gesture recognition approaches is given in this paper. We shall review methods of static hand posture and temporal gesture recognition. Several application systems of gesture recognition are also described in this paper. We conclude with some thoughts about future research directions.
Limitations of the research: """
FEWSHOT_STRM_2A_SCIDOCS = """The review focuses on existing vision-based gesture recognition approaches but does not provide a unified framework or methodology to address the challenges identified in the field. While it surveys static hand posture and temporal gesture recognition methods, it does not delve deeply into the comparative performance or limitations of specific algorithms, leaving gaps in understanding their practical applicability. The paper primarily emphasizes the technological aspects of gesture recognition but lacks substantial discussion on real-world challenges, such as varying lighting conditions, occlusions, and diverse user profiles, which are critical for deploying these systems in practical scenarios.

Interdisciplinary studies are acknowledged as essential for gesture recognition, but the review does not explore how these disciplines can be effectively integrated, nor does it highlight gaps in collaboration between fields like computer vision, human-computer interaction, and cognitive science. The application systems described in the paper are presented without extensive evaluation of their scalability, robustness, or usability in dynamic environments. Additionally, there is limited discussion on the ethical, privacy, and accessibility issues surrounding gesture recognition, which are increasingly relevant in modern applications. The conclusions on future research directions are broad, leaving room for more precise recommendations on addressing the identified limitations in current approaches."""
# FEWSHOT_STRM_2H_SCIDOCS = """Example 2:
# Document: Vision-Based Gesture Recognition: A Review The use of gesture as a natural interface serves as a motivating force for research in modeling, analyzing and recognition of gestures. In particular, human computer intelligent interaction needs vision-based gesture recognition, which involves many interdisciplinary studies. A survey on recent vision-based gesture recognition approaches is given in this paper. We shall review methods of static hand posture and temporal gesture recognition. Several application systems of gesture recognition are also described in this paper. We conclude with some thoughts about future research directions.
# Brainstorm the most plausible future research topic: """
# FEWSHOT_STRM_2A_SCIDOCS = """Future research could focus on designing robust, real-time vision-based gesture recognition systems tailored for dynamic, real-world environments. This includes investigating algorithms capable of adapting to varying lighting conditions, occlusions, and diverse user profiles, ensuring reliable recognition in practical scenarios. A promising direction involves integrating gesture recognition with other input modalities, such as speech recognition or eye tracking, to create seamless, context-aware human-computer interaction systems. Efforts could also concentrate on developing efficient gesture recognition models optimized for edge devices, enabling real-time processing on portable and wearable platforms. Addressing cultural and contextual variations in gesture interpretation would enhance the system's applicability across diverse user groups. Furthermore, extending gesture recognition capabilities to immersive technologies, such as virtual and augmented reality, could revolutionize user experiences by providing intuitive and natural interaction methods."""
FEWSHOT_STRM_2H_ARGUANA = """Example 2:
Document: Within cities land grabbing is a myth. A number of cases shown as political land-grabbing and rent-seeking are misrepresented, and misunderstood. Difficulties remain in defining what is a land grab and the extent of which the state, and politics, are involved in land speculations.  The media coverage of evictions in Mogadishu showcase the myth and hyperbole surrounding African politics and evictions. The government are entitled to reclaim land and reform it for public use [1] .  [1] See BBC News (2013) for full debate, whereby Mohammed Yusuf, an Official at Mogadishu City, defends the eviction.
Main Argument and Supporting Evidence: """
FEWSHOT_STRM_2A_ARGUANA = """Main Argument:
The concept of land grabbing within cities is largely a myth, often misrepresented and misunderstood, particularly in the context of political land-grabbing and rent-seeking.

Supporting Evidence:
Cases labeled as political land-grabbing and rent-seeking are frequently exaggerated or inaccurately portrayed. The media coverage of evictions in Mogadishu exemplifies this myth, contributing to hyperbolic narratives about African politics and evictions. Additionally, the government has the legitimate authority to reclaim and reform land for public use, as highlighted by Mohammed Yusuf, an official at Mogadishu City, during a debate covered by BBC News in 2013. This demonstrates that such evictions are often part of lawful urban planning and reform rather than acts of exploitation."""
FEWSHOT_STRM_2H_QUORA = """Example 2:
Question: How can I learn about the basics of computer and information security?
Rewritten Questions: """
FEWSHOT_STRM_2A_QUORA = """What are the best ways to learn the fundamentals of computer and information security?
How can a beginner start understanding the basics of cybersecurity?
Where should I begin to learn about computer and information security concepts?
What resources are recommended for learning the essentials of information security?
How can I develop a foundational knowledge of cybersecurity?
What are the key topics to focus on when starting with computer and information security?
How can someone new to the field learn about information security basics?
What is the best approach to understanding computer security for beginners?
How can I start building my knowledge in the field of cybersecurity?
What introductory materials or courses are available for learning information security?"""

FEWSHOT_STRM_2H_MAP = {
    "scidocs": FEWSHOT_STRM_2H_SCIDOCS,
    "arguana": FEWSHOT_STRM_2H_ARGUANA,
    "quora": FEWSHOT_STRM_2H_QUORA,
}
FEWSHOT_STRM_2A_MAP = {
    "scidocs": FEWSHOT_STRM_2A_SCIDOCS,
    "arguana": FEWSHOT_STRM_2A_ARGUANA,
    "quora": FEWSHOT_STRM_2A_QUORA,
}

FEWSHOT_STRM_3H_SCIDOCS = """Example 3:
Document: Review of Inflatable Booms for Deployable Space Structures : Packing and Rigidization Inflatable structures offer the potential of compactly stowing lightweight structures, which assume a fully deployed state in space. An important category of space inflatables are cylindrical booms, which may form the structural members of trusses or the support structure for solar sails. Two critical and interdependent aspects of designing inflatable cylindrical booms for space applications are i) packaging methods that enable compact stowage and ensure reliable deployment, and ii) rigidization techniques that provide long-term structural ridigity after deployment. The vast literature in these two fields is summarized to establish the state of the art.
Limitations of the research: """
FEWSHOT_STRM_3A_SCIDOCS = """The review focuses on summarizing the state of the art in packaging and rigidization techniques for inflatable cylindrical booms but does not propose new methodologies or address existing gaps in the field. While it highlights the importance of compact stowage and reliable deployment, it does not explore in depth how these methods perform under diverse space conditions, such as extreme temperatures, radiation exposure, or microgravity effects over time. The review lacks a comprehensive analysis of the trade-offs between packaging efficiency and structural integrity after deployment, which is critical for practical applications.

Additionally, the discussion on rigidization techniques is broad and does not assess the long-term durability or potential failure modes of these methods in space environments. The literature summarized may also have limitations in scope, potentially overlooking emerging materials or innovative technologies that could address existing challenges. There is limited emphasis on scalability for larger structures or adaptability for multi-functional designs that integrate power generation or thermal management. Finally, while the review establishes the current state of knowledge, it does not clearly define pathways for future research or address interdisciplinary approaches that could accelerate advancements in this field."""
# FEWSHOT_STRM_3H_SCIDOCS = """Example 3:
# Document: Review of Inflatable Booms for Deployable Space Structures : Packing and Rigidization Inflatable structures offer the potential of compactly stowing lightweight structures, which assume a fully deployed state in space. An important category of space inflatables are cylindrical booms, which may form the structural members of trusses or the support structure for solar sails. Two critical and interdependent aspects of designing inflatable cylindrical booms for space applications are i) packaging methods that enable compact stowage and ensure reliable deployment, and ii) rigidization techniques that provide long-term structural ridigity after deployment. The vast literature in these two fields is summarized to establish the state of the art.
# Brainstorm the most plausible future research topic: """
# FEWSHOT_STRM_3A_SCIDOCS = """Future research could focus on developing advanced materials and autonomous deployment systems for inflatable space structures, addressing challenges in packaging, deployment reliability, and long-term structural rigidity. This may include exploring novel composite materials with enhanced rigidity-to-weight ratios and self-healing properties to improve durability and resilience in extreme space environments. Autonomous systems leveraging artificial intelligence and machine learning could be designed to monitor and control deployment processes, ensuring optimal performance and reducing the risk of failure. Additionally, investigating multifunctional designs that integrate power generation, thermal management, and radiation shielding into the inflatable structures could enhance their utility and efficiency for long-duration missions. These innovations would significantly expand the applicability of inflatable space structures for applications such as large-scale solar sails, expandable habitats, and modular space infrastructure."""
FEWSHOT_STRM_3H_ARGUANA = """Example 3:
Document: Home-schooling is not the best option for exceptional students. The state does not ignore or abandon individuals that have special needs and those with special needs are those that most need the state's enormous resources to focus on their requirements. Once a student has needs of such a magnitude that demands it, they are educated in special schools specifically intended to help them, with staff trained to possess skills beyond that of a parent's instinct. Even if it were the case that home-schooling is better for the specific needs of exceptional students, the benefits of education in a wider context override the objection to class-based education. The experience of growing up alongside less and more able students produces individuals with greater understanding of their society1. 1'Teacher perceptions of mainstreaming/inclusion, 1958-1995: a research synthesis' Scruggs, Thomas E. Mastropieri, Margo A. Exceptional Children (1996)
Main Argument and Supporting Evidence: """
FEWSHOT_STRM_3A_ARGUANA = """Main Argument:
Home-schooling is not the best option for exceptional students because state-supported special education provides superior resources, trained staff, and a broader social learning environment that better supports their development.

Supporting Evidence:
The state offers substantial resources to address the needs of exceptional students, including specialized schools with staff trained in skills beyond what parents typically possess. These schools are specifically designed to cater to the unique requirements of students with significant needs. Even if home-schooling could better meet some individual needs, the broader benefits of classroom-based education outweigh these advantages. Interaction with peers of varying abilities fosters a deeper understanding of society and promotes social development. Research by Scruggs and Mastropieri (1996) supports this, emphasizing the value of mainstreaming and inclusion in educational contexts."""
FEWSHOT_STRM_3H_QUORA = """Example 3:
Question: When Obama leaves office, will he give up the @POTUS account on Twitter?
Rewritten Questions: """
FEWSHOT_STRM_3A_QUORA = """What happens to the @POTUS Twitter account when Obama leaves office?
Will Obama transfer control of the @POTUS Twitter account after his presidency?
Does the @POTUS account stay with the office or with the individual president?
How is the @POTUS Twitter account managed after a president leaves office?
Will Obama retain access to the @POTUS Twitter account once his term ends?
What is the protocol for the @POTUS account when a new president takes office?
Who takes over the @POTUS Twitter account after Obama’s presidency?
Does the @POTUS account transition to the next president automatically?
Is the @POTUS Twitter account reassigned to the new administration after Obama?
How does the ownership of the @POTUS account change with each presidency?"""

FEWSHOT_STRM_3H_MAP = {
    "scidocs": FEWSHOT_STRM_3H_SCIDOCS,
    "arguana": FEWSHOT_STRM_3H_ARGUANA,
    "quora": FEWSHOT_STRM_3H_QUORA,
}
FEWSHOT_STRM_3A_MAP = {
    "scidocs": FEWSHOT_STRM_3A_SCIDOCS,
    "arguana": FEWSHOT_STRM_3A_ARGUANA,
    "quora": FEWSHOT_STRM_3A_QUORA,
}

FEWSHOT_STRM_4H_SCIDOCS = """Example 4:
Document: {document}
Brainstorm the most plausible future research topic: """
FEWSHOT_STRM_4H_ARGUANA = """Example 4:
Document: {document}
Main Argument and Supporting Evidence: """
FEWSHOT_STRM_4H_QUORA = """Example 4:
Question: {document}
Rewritten Questions: """

FEWSHOT_STRM_4H_MAP = {
    "scidocs": FEWSHOT_STRM_4H_SCIDOCS,
    "arguana": FEWSHOT_STRM_4H_ARGUANA,
    "quora": FEWSHOT_STRM_4H_QUORA,
}


async def streamline_fewshot_gen_async_dataset(doc, client, dataset):
    prompt = FEWSHOT_STRM_4H_MAP[dataset].format(document=doc, query="").rstrip()
    chat_prompt_with_inst = ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=FEWSHOT_STRM_1H_MAP[dataset]),
            AIMessage(content=FEWSHOT_STRM_1A_MAP[dataset]),
            HumanMessage(content=FEWSHOT_STRM_2H_MAP[dataset]),
            AIMessage(content=FEWSHOT_STRM_2A_MAP[dataset]),
            HumanMessage(content=FEWSHOT_STRM_3H_MAP[dataset]),
            AIMessage(content=FEWSHOT_STRM_3A_MAP[dataset]),
            HumanMessage(content=prompt),
        ]
    )
    chain = chat_prompt_with_inst | client
    result = await chain.ainvoke({})
    content = (
        result.content.replace(prompt, "")
        .split("Example 5:")[0]
        .strip()
        .split(
            "In each of these examples, the AI model is trained on a large corpus of text data,"
        )[0]
        .strip()
        .split("Answer:")[0]
        .strip()
        .split("\n")[0]
        .strip()
        .split("?")[0]
        .strip()
    )
    return content


FEWSHOT_STRM_QGEN_1H_SCIDOCS = f"""Example 1:
Limitations of the research: {FEWSHOT_STRM_1A_SCIDOCS}
Future Research Title: """
FEWSHOT_STRM_QGEN_1A_SCIDOCS = "Enhancing Thermal Imaging-Based Deception Detection: Toward Multimodal, Generalizable, and Ethical Frameworks"
# FEWSHOT_STRM_QGEN_1H_SCIDOCS = f"""Example 1:
# Future Research Explain: {FEWSHOT_STRM_1A_SCIDOCS}
# Future Research Title: """
FEWSHOT_STRM_QGEN_1A_SCIDOCS = "Multimodal Real-Time Deception Detection: Integrating Thermal Imaging with Physiological and Behavioral Analysis"
FEWSHOT_STRM_QGEN_2H_SCIDOCS = f"""Example 2:
Limitations of the research: {FEWSHOT_STRM_2A_SCIDOCS}
Future Research Title: """
FEWSHOT_STRM_QGEN_2A_SCIDOCS = "Advancing Vision-Based Gesture Recognition: Integrating Real-World Challenges, Interdisciplinary Approaches, and Ethical Considerations"
# FEWSHOT_STRM_QGEN_2H_SCIDOCS = f"""Example 2:
# Future Research Explain: {FEWSHOT_STRM_2A_SCIDOCS}
# Future Research Title: """
# FEWSHOT_STRM_QGEN_2A_SCIDOCS = "Adaptive and Multimodal Gesture Recognition Systems for Real-World and Immersive Applications"
FEWSHOT_STRM_QGEN_3H_SCIDOCS = f"""Example 3:
Limitations of the research: {FEWSHOT_STRM_3A_SCIDOCS}
Future Research Title: """
FEWSHOT_STRM_QGEN_3A_SCIDOCS = "Innovative Strategies for Inflatable Space Structures: Addressing Durability, Scalability, and Multifunctionality"
# FEWSHOT_STRM_QGEN_3H_SCIDOCS = f"""Example 3:
# Future Research Explain: {FEWSHOT_STRM_3A_SCIDOCS}
# Future Research Title: """
# FEWSHOT_STRM_QGEN_3A_SCIDOCS = "Innovative Materials and Autonomous Systems for Next-Generation Inflatable Space Structures"
FEWSHOT_STRM_QGEN_4H_SCIDOCS = """Example 4:
Limitations of the research: {document}
Future Research Title: """
# FEWSHOT_STRM_QGEN_4H_SCIDOCS = """Example 4:
# Future Research Explain: {document}
# Future Research Title: """
FEWSHOT_STRM_QGEN_1H_ARGUANA = f"""Example 1:
Main Argument and Supporting Evidence: {FEWSHOT_STRM_1A_ARGUANA}
Counter Argument: """
FEWSHOT_STRM_QGEN_1A_ARGUANA = "A graduate tax can be viewed as a fair mechanism to ensure that individuals who benefit financially from higher education contribute proportionally to its costs. By linking repayment to income levels, the tax ensures that those with higher earnings contribute more, while those with lower earnings or career paths in less lucrative fields face a lighter burden. This system could provide sustainable funding for universities, enabling them to maintain and improve the quality of education. Furthermore, the graduate tax could potentially expand access to higher education by reducing the upfront costs for students, making university more financially accessible regardless of socioeconomic background. Critics of the commoditization argument may also assert that combining education with employability goals does not diminish the intrinsic value of learning but rather aligns universities with the practical realities of modern economies."
FEWSHOT_STRM_QGEN_2H_ARGUANA = f"""Example 2:
Main Argument and Supporting Evidence: {FEWSHOT_STRM_2A_ARGUANA}
Counter Argument: """
FEWSHOT_STRM_QGEN_2A_ARGUANA = """While some cases of land grabbing may be exaggerated or misunderstood, the phenomenon is not entirely a myth. In many urban contexts, evictions and land reallocations often disproportionately affect marginalized communities, leading to displacement without adequate compensation or relocation plans. Critics argue that the label of "lawful urban planning" can sometimes be used to mask exploitative practices, particularly when governments or private entities prioritize economic development or political interests over the rights and welfare of vulnerable populations. In cities like Mogadishu, the lack of transparency and accountability in land reclamation processes can contribute to perceptions of unfairness and exploitation. Moreover, the media's coverage of such incidents often highlights the real human cost of evictions, which cannot be dismissed as mere hyperbole. This suggests that while land reclamation may be lawful in principle, its implementation can have negative social and ethical implications."""
FEWSHOT_STRM_QGEN_3H_ARGUANA = f"""Example 3:
Main Argument and Supporting Evidence: {FEWSHOT_STRM_3A_ARGUANA}
Counter Argument: """
FEWSHOT_STRM_QGEN_3A_ARGUANA = "Home-schooling can provide a highly personalized and flexible learning environment that may better address the unique needs of exceptional students. Unlike state-supported special education, home-schooling allows parents to tailor the curriculum, pace, and teaching methods to suit their child's strengths and challenges, fostering a more supportive and adaptive educational experience. Critics argue that the one-size-fits-all approach of specialized schools may not adequately address the individuality of some students, potentially leaving their needs unmet. Furthermore, home-schooling can offer a safe and comfortable learning space for students who struggle with social or sensory challenges, which can sometimes be exacerbated in a classroom setting. While interaction with peers is important, socialization opportunities can still be provided through community groups, extracurricular activities, or home-schooling cooperatives, offering a balance between personalized education and social development."
FEWSHOT_STRM_QGEN_4H_ARGUANA = """Example 4:
Main Argument and Supporting Evidence: {document}
Counter Argument: """
FEWSHOT_STRM_QGEN_1H_QUORA = f"""Example 1:
Rewritten Questions: {FEWSHOT_STRM_1A_QUORA}
Best Question: """
FEWSHOT_STRM_QGEN_1A_QUORA = (
    "What are the indicators that you may have reached spiritual enlightenment?"
)
FEWSHOT_STRM_QGEN_2H_QUORA = f"""Example 2:
Rewritten Questions: {FEWSHOT_STRM_2A_QUORA}
Best Question: """
FEWSHOT_STRM_QGEN_2A_QUORA = "What are the best ways to learn the fundamentals of computer and information security?"
FEWSHOT_STRM_QGEN_3H_QUORA = f"""Example 3:
Rewritten Questions: {FEWSHOT_STRM_3A_QUORA}
Best Question: """
FEWSHOT_STRM_QGEN_3A_QUORA = (
    "What happens to the @POTUS Twitter account when Obama leaves office?"
)
FEWSHOT_STRM_QGEN_4H_QUORA = """Example 4:
Rewritten Questions: {document}
Best Question: """


FEWSHOT_STRM_QGEN_1H_MAP = {
    "scidocs": FEWSHOT_STRM_QGEN_1H_SCIDOCS,
    "arguana": FEWSHOT_STRM_QGEN_1H_ARGUANA,
    "quora": FEWSHOT_STRM_QGEN_1H_QUORA,
}
FEWSHOT_STRM_QGEN_1A_MAP = {
    "scidocs": FEWSHOT_STRM_QGEN_1A_SCIDOCS,
    "arguana": FEWSHOT_STRM_QGEN_1A_ARGUANA,
    "quora": FEWSHOT_STRM_QGEN_1A_QUORA,
}
FEWSHOT_STRM_QGEN_2H_MAP = {
    "scidocs": FEWSHOT_STRM_QGEN_2H_SCIDOCS,
    "arguana": FEWSHOT_STRM_QGEN_2H_ARGUANA,
    "quora": FEWSHOT_STRM_QGEN_2H_QUORA,
}
FEWSHOT_STRM_QGEN_2A_MAP = {
    "scidocs": FEWSHOT_STRM_QGEN_2A_SCIDOCS,
    "arguana": FEWSHOT_STRM_QGEN_2A_ARGUANA,
    "quora": FEWSHOT_STRM_QGEN_2A_QUORA,
}
FEWSHOT_STRM_QGEN_3H_MAP = {
    "scidocs": FEWSHOT_STRM_QGEN_3H_SCIDOCS,
    "arguana": FEWSHOT_STRM_QGEN_3H_ARGUANA,
    "quora": FEWSHOT_STRM_QGEN_3H_QUORA,
}
FEWSHOT_STRM_QGEN_3A_MAP = {
    "scidocs": FEWSHOT_STRM_QGEN_3A_SCIDOCS,
    "arguana": FEWSHOT_STRM_QGEN_3A_ARGUANA,
    "quora": FEWSHOT_STRM_QGEN_3A_QUORA,
}
FEWSHOT_STRM_QGEN_4H_MAP = {
    "scidocs": FEWSHOT_STRM_QGEN_4H_SCIDOCS,
    "arguana": FEWSHOT_STRM_QGEN_4H_ARGUANA,
    "quora": FEWSHOT_STRM_QGEN_4H_QUORA,
}


async def streamline_duqgen_async_dataset(doc, client, dataset):
    prompt = FEWSHOT_STRM_QGEN_4H_MAP[dataset].format(document=doc, query="").rstrip()
    chat_prompt_with_inst = ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=FEWSHOT_STRM_QGEN_1H_MAP[dataset]),
            AIMessage(content=FEWSHOT_STRM_QGEN_1A_MAP[dataset]),
            HumanMessage(content=FEWSHOT_STRM_QGEN_2H_MAP[dataset]),
            AIMessage(content=FEWSHOT_STRM_QGEN_2A_MAP[dataset]),
            HumanMessage(content=FEWSHOT_STRM_QGEN_3H_MAP[dataset]),
            AIMessage(content=FEWSHOT_STRM_QGEN_3A_MAP[dataset]),
            HumanMessage(content=prompt),
        ]
    )
    chain = chat_prompt_with_inst | client
    result = await chain.ainvoke({})
    content = (
        result.content.replace(prompt, "")
        .split("Example 5:")[0]
        .strip()
        .split(
            "In each of these examples, the AI model is trained on a large corpus of text data,"
        )[0]
        .strip()
        .split("Answer:")[0]
        .strip()
        .split("\n")[0]
        .strip()
        .split("?")[0]
        .strip()
    )
    return content


CLARIFY_INSTRUCTION = """
Please rewrite the following document to enhance its readability and incorporate a wider range of vocabulary to address potential vocabulary mismatch issues. Summarize sections if necessary to improve clarity, but leave the text unchanged if it is already clear and well-written.
"""


async def clarify_summary_gen_async(doc, client):
    chat_prompt_with_inst = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=CLARIFY_INSTRUCTION),
            HumanMessagePromptTemplate.from_template("{user_message}"),
        ]
    )
    user_prompt_pos = {"user_message": f"Document: {doc}\n\nOutput:"}
    chain = chat_prompt_with_inst | client
    result = await chain.ainvoke(user_prompt_pos)
    content = remove_intro_line(result.content)
    return content


KEYWORD_INSTRUCTION = """
Extract up to three keywords that appears in the document and best represent the main topics or themes of the following document:
"""


async def keyword_gen_async(doc, client):
    prompt = FEWSHOT_4H_KWGEN.format(document=doc, keywords="", query="").rstrip()
    chat_prompt_with_inst = ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=FEWSHOT_1H_KWGEN),
            AIMessage(content=FEWSHOT_1A_KWGEN),
            HumanMessage(content=FEWSHOT_2H_KWGEN),
            AIMessage(content=FEWSHOT_2A_KWGEN),
            HumanMessage(content=FEWSHOT_3H_KWGEN),
            AIMessage(content=FEWSHOT_3A_KWGEN),
            HumanMessage(content=prompt),
        ]
    )
    chain = chat_prompt_with_inst | client
    result = await chain.ainvoke({})
    content = remove_intro_line(result.content)
    return content


EXTRACT_INSTRUCTION = """You are an efficient search assistant. I will provide you the document.
You will begin by examining the documents, you will extract the key sentences from the document that contribute to their main argument."""

EXTRACT_FEWSHOT_H = """Document: Whoever pays the most, gets the produce. The longer it sits, starts going bad, the lower the value goes. For walmart to sell at their prices, they need to buy as cheap as possible.   Most big juice companies don't have their own farms, they buy for the lowest price to make a profit, they actually just buy the fruit as its less of a risk. The one's they buy definitely do not look like whats on the juice box. If you dig into it, there is more info on the internet that shows these kinds of things. The food industry is quiet bad. \n\nExtracted Key Sentences: """
EXTRACT_FEWSHOT_A = """* Whoever pays the most, gets the produce. The longer it sits, starts going bad, the lower the value goes.
* They need to buy as cheap as possible.
* They actually just buy the fruit as its less of a risk.
* The food industry is quiet bad."""


async def summary_extract_async(doc, client):
    chat_prompt_with_inst = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=EXTRACT_INSTRUCTION),
            HumanMessage(content=EXTRACT_FEWSHOT_H),
            AIMessage(content=EXTRACT_FEWSHOT_A),
            HumanMessagePromptTemplate.from_template("{user_message}"),
        ]
    )

    user_prompt_pos = {"user_message": f"Document: {doc}\n\nExtracted Key Sentences: "}
    chain = chat_prompt_with_inst | client
    result = await chain.ainvoke(user_prompt_pos)
    content = " ".join(
        [
            e.replace("* ", "")
            for e in result.content.split("Here are the extracted key sentences:")[-1]
            .strip()
            .split("\n")
        ]
    ).strip()
    return content


from snowflake.cortex import ConversationMessage
from typing import List

# --------------------------------------------------
async def duqgen_dataset_snow(document: str, client, dataset: str) -> str:
    user_turn = FEWSHOT_4H_MAP[dataset].format(document=document, query="").rstrip()

    messages: List[ConversationMessage] = [
        ConversationMessage(role="user", content=FEWSHOT_1H_MAP[dataset]),
        ConversationMessage(role="assistant", content=FEWSHOT_1A_MAP[dataset]),
        ConversationMessage(role="user", content=FEWSHOT_2H_MAP[dataset]),
        ConversationMessage(role="assistant", content=FEWSHOT_2A_MAP[dataset]),
        ConversationMessage(role="user", content=FEWSHOT_3H_MAP[dataset]),
        ConversationMessage(role="assistant", content=FEWSHOT_3A_MAP[dataset]),
        ConversationMessage(role="user", content=user_turn),
    ]

    raw = await client.agenerate(messages, max_tokens=100, temperature=0.8)

    # --- post‑processing stays exactly as before ---
    cleaned = (
        raw.replace(user_turn, "")
        .split("Example 5:")[0]
        .split("In each of these examples")[0]
        .split("Answer:")[0]
        .strip()
        .split("\n")[0]
        .split("?")[0]
        .strip()
    )
    return cleaned


async def per_sents_dataset_snow(document: str, sentence: str, client, dataset: str) -> str:
    user_turn = FEWSHOT_4H_SENT_MAP[dataset].format(
        document=document, sentence=sentence, query=""
    ).rstrip()

    messages: List[ConversationMessage] = [
        ConversationMessage(role="user", content=FEWSHOT_1H_SENT_MAP[dataset]),
        ConversationMessage(role="assistant", content=FEWSHOT_1A_MAP[dataset]),
        ConversationMessage(role="user", content=FEWSHOT_2H_SENT_MAP[dataset]),
        ConversationMessage(role="assistant", content=FEWSHOT_2A_MAP[dataset]),
        ConversationMessage(role="user", content=FEWSHOT_3H_SENT_MAP[dataset]),
        ConversationMessage(role="assistant", content=FEWSHOT_3A_MAP[dataset]),
        ConversationMessage(role="user", content=user_turn),
    ]

    raw = await client.agenerate(messages, max_tokens=100, temperature=0.8)

    cleaned = (
        raw.replace(user_turn, "")
        .split("Example 5:")[0]
        .split("In each of these examples")[0]
        .split("Answer:")[0]
        .strip()
        .split("\n")[0]
        .split("?")[0]
        .strip()
    )
    return cleaned

async def per_sents_dataset(doc, sentence, client, dataset):
    prompt = FEWSHOT_4H_SENT_MAP[dataset].format(document=doc, sentence=sentence, query="").rstrip()
    chat_prompt_with_inst = ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=FEWSHOT_1H_SENT_MAP[dataset]),
            AIMessage(content=FEWSHOT_1A_MAP[dataset]),
            HumanMessage(content=FEWSHOT_2H_SENT_MAP[dataset]),
            AIMessage(content=FEWSHOT_2A_MAP[dataset]),
            HumanMessage(content=FEWSHOT_3H_SENT_MAP[dataset]),
            AIMessage(content=FEWSHOT_3A_MAP[dataset]),
            HumanMessage(content=prompt),
        ]
    )
    chain = chat_prompt_with_inst | client
    result = await chain.ainvoke({})
    content = (
        result.content.replace(prompt, "")
        .split("Example 5:")[0]
        .strip()
        .split(
            "In each of these examples, the AI model is trained on a large corpus of text data,"
        )[0]
        .strip()
        .split("Answer:")[0]
        .strip()
        .split("\n")[0]
        .strip()
        .split("?")[0]
        .strip()
    )
    return content


async def per_sents_dataset_qps(doc, sentence, client, dataset, qps):
    def is_similar(a, b, threshold=0.9):
        return SequenceMatcher(None, a, b).ratio() > threshold

    def postprocess(result):
        return (result.content.replace(prompt, "")
        .split("Example 5:")[0]
        .strip()
        .split(
            "In each of these examples, the AI model is trained on a large corpus of text data,"
        )[0]
        .strip()
        .split("Answer:")[0]
        .strip()
        .split("\n")[0]
        .strip()
        .split("?")[0]
        .strip())

    prompt = FEWSHOT_4H_SENT_MAP[dataset].format(document=doc, sentence=sentence, query="").rstrip()
    chat_prompt_with_inst = ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=FEWSHOT_1H_SENT_MAP[dataset]),
            AIMessage(content=FEWSHOT_1A_MAP[dataset]),
            HumanMessage(content=FEWSHOT_2H_SENT_MAP[dataset]),
            AIMessage(content=FEWSHOT_2A_MAP[dataset]),
            HumanMessage(content=FEWSHOT_3H_SENT_MAP[dataset]),
            AIMessage(content=FEWSHOT_3A_MAP[dataset]),
            HumanMessage(content=prompt),
        ]
    )
    chain = chat_prompt_with_inst | client
    target_left = qps
    generated_queries = []
    max_retry = 5
    while max_retry > 0:
        results = await chain.abatch([{} for _ in range(target_left)])
        queries = [postprocess(result) for result in results]
        for query in queries:
            for genq in generated_queries:
                if is_similar(query, genq):
                    break
            generated_queries.append(query)
        target_left = qps - len(generated_queries)
        if target_left == 0:
            break
    if max_retry == 0:
        raise ValueError("Failed to generate queries within the retry limit.")
    return generated_queries


async def duqgen_dataset(doc, client, dataset):
    prompt = FEWSHOT_4H_MAP[dataset].format(document=doc, query="").rstrip()
    chat_prompt_with_inst = ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=FEWSHOT_1H_MAP[dataset]),
            AIMessage(content=FEWSHOT_1A_MAP[dataset]),
            HumanMessage(content=FEWSHOT_2H_MAP[dataset]),
            AIMessage(content=FEWSHOT_2A_MAP[dataset]),
            HumanMessage(content=FEWSHOT_3H_MAP[dataset]),
            AIMessage(content=FEWSHOT_3A_MAP[dataset]),
            HumanMessage(content=prompt),
        ]
    )
    chain = chat_prompt_with_inst | client
    result = await chain.ainvoke({})
    content = (
        result.content.replace(prompt, "")
        .split("Example 5:")[0]
        .strip()
        .split(
            "In each of these examples, the AI model is trained on a large corpus of text data,"
        )[0]
        .strip()
        .split("Answer:")[0]
        .strip()
        .split("\n")[0]
        .strip()
        .split("?")[0]
        .strip()
    )
    return content

async def duqgen(doc, client):
    prompt = FEWSHOT_4H.format(document=doc, query="").rstrip()
    chat_prompt_with_inst = ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=FEWSHOT_1H),
            AIMessage(content=FEWSHOT_1A),
            HumanMessage(content=FEWSHOT_2H),
            AIMessage(content=FEWSHOT_2A),
            HumanMessage(content=FEWSHOT_3H),
            AIMessage(content=FEWSHOT_3A),
            HumanMessage(content=prompt),
        ]
    )
    chain = chat_prompt_with_inst | client
    result = await chain.ainvoke({})
    content = (
        result.content.replace(prompt, "")
        .split("Example 5:")[0]
        .strip()
        .split(
            "In each of these examples, the AI model is trained on a large corpus of text data,"
        )[0]
        .strip()
        .split("Answer:")[0]
        .strip()
        .split("\n")[0]
        .strip()
        .split("?")[0]
        .strip()
    )
    return content


async def duqgen_summary(doc, client):
    prompt = FEWSHOT_4H_SUMMARY.format(summary=doc, query="").rstrip()
    chat_prompt_with_inst = ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=FEWSHOT_1H_SUMMARY),
            AIMessage(content=FEWSHOT_1A_SUMMARY),
            HumanMessage(content=FEWSHOT_2H_SUMMARY),
            AIMessage(content=FEWSHOT_2A_SUMMARY),
            HumanMessage(content=FEWSHOT_3H_SUMMARY),
            AIMessage(content=FEWSHOT_3A_SUMMARY),
            HumanMessage(content=prompt),
        ]
    )
    chain = chat_prompt_with_inst | client
    result = await chain.ainvoke({})
    content = (
        result.content.replace(prompt, "")
        .split("Example 5:")[0]
        .strip()
        .split(
            "In each of these examples, the AI model is trained on a large corpus of text data,"
        )[0]
        .strip()
        .split("Answer:")[0]
        .strip()
        .split("\n")[0]
        .strip()
        .split("?")[0]
        .strip()
    )
    return content


async def duqgen_streamlined(doc, client):
    prompt = FEWSHOT_4H_STREAMLINED.format(document=doc, query="").rstrip()
    chat_prompt_with_inst = ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=FEWSHOT_1H_STREAMLINED),
            AIMessage(content=FEWSHOT_1A_STREAMLINED),
            HumanMessage(content=FEWSHOT_2H_STREAMLINED),
            AIMessage(content=FEWSHOT_2A_STREAMLINED),
            HumanMessage(content=FEWSHOT_3H_STREAMLINED),
            AIMessage(content=FEWSHOT_3A_STREAMLINED),
            HumanMessage(content=prompt),
        ]
    )
    chain = chat_prompt_with_inst | client
    result = await chain.ainvoke({})
    content = (
        result.content.replace(prompt, "")
        .split("Example 5:")[0]
        .strip()
        .split(
            "In each of these examples, the AI model is trained on a large corpus of text data,"
        )[0]
        .strip()
        .split("Answer:")[0]
        .strip()
        .split("\n")[0]
        .strip()
        .split("?")[0]
        .strip()
    )
    return content


async def duqgen_keyword(doc, keywords, client):
    prompt = FEWSHOT_4H_KW.format(document=doc, keywords=keywords, query="").rstrip()
    chat_prompt_with_inst = ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=FEWSHOT_1H_KW),
            AIMessage(content=FEWSHOT_1A),
            HumanMessage(content=FEWSHOT_2H_KW),
            AIMessage(content=FEWSHOT_2A),
            HumanMessage(content=FEWSHOT_3H_KW),
            AIMessage(content=FEWSHOT_3A),
            HumanMessage(content=prompt),
        ]
    )
    chain = chat_prompt_with_inst | client
    result = await chain.ainvoke({})
    content = (
        result.content.replace(prompt, "")
        .split("Example 5:")[0]
        .strip()
        .split(
            "In each of these examples, the AI model is trained on a large corpus of text data,"
        )[0]
        .strip()
        .split("Answer:")[0]
        .strip()
        .split("\n")[0]
        .strip()
        .split("?")[0]
        .strip()
    )
    return content


# async def duqgen_summary(doc, summary, client):
#     prompt = FEWSHOT_4H_SM.format(document=doc, summary=summary, query="").rstrip()
#     chat_prompt_with_inst = ChatPromptTemplate.from_messages(
#         [
#             HumanMessage(content=FEWSHOT_1H_SM),
#             AIMessage(content=FEWSHOT_1A),
#             HumanMessage(content=FEWSHOT_2H_SM),
#             AIMessage(content=FEWSHOT_2A),
#             HumanMessage(content=FEWSHOT_3H_SM),
#             AIMessage(content=FEWSHOT_3A),
#             HumanMessage(content=prompt),
#         ]
#     )
#     chain = chat_prompt_with_inst | client
#     result = await chain.ainvoke({})
#     content = (
#         result.content.replace(prompt, "")
#         .split("Example 5:")[0]
#         .strip()
#         .split(
#             "In each of these examples, the AI model is trained on a large corpus of text data,"
#         )[0]
#         .strip()
#         .split("Answer:")[0]
#         .strip()
#         .split("\n")[0]
#         .strip()
#         .split("?")[0]
#         .strip()
#     )
#     return content


PICK_SENT_INSTRUCTION = """Given the following document, extract key sentences that grounds the possible question generation. Sentences should be ordered by their importance and relevance to the document's content."""
PICK_SENT_FEWSHOT_H = """Document: Breathing is an active process requiring the contraction of skeletal muscles. The primary muscles of respiration include the external intercostal muscles (located between the ribs) and the diaphragm (a sheet of muscle located between the thoracic & abdominal cavities). Breathing is a physical process consisting of inhalation, exhalation, and relaxation. Inhalation is an active process while exhalation is passive. Breathing involves two stages known as ventilation and gas exchange. Ventilation is the movement of the air in and out of the lungs. Exhalation—Carbon dioxide is pushed out of your body through the nose or mouth when your diaphragm contracts. At the site of the alveoli, the oxygen goes into the oxygen-poor blood that is coming from all over the body. The carbon dioxide is then forced out and breathed out. Answered by The Community. Making the world better, one answer at a time. Inhalation is a process that relies on contraction of muscles while exhalation is a process that is usually—not always—passive because it relies on the relaxation of muscles. However, when you speak, sing, or blow out a candle, the muscles between the ribs and abs contract, making it active. An active process is one that is carried out at the expense of energy, and a passive process occurs spontaneously in nature. For example, consider two solutions separated by a membrane; one side has a higher concentration of glucose than the other. • Inhalation is an active process, whereas exhalation is a passive process. • Exhalation occurs followed by inhalation. • The diaphragm and intercostal muscles contract during inhalation, while they relax during exhalation. Inhalation is an active process in which a person takes air into the body through the mouth and nose and pushes the air into the lungs. Inhalation is controlled by the brain. During the process of inhalation, the diaphragm and intercostal muscle contractions cause the thoracic cavity to enlarge. This creates a slight vacuum condition due to decreasing air pressure in the lungs. Due to the pressure gradient between the atmosphere and thoracic cavity, air moves into the lungs via the trachea.

Extracted Key Sentences: """
PICK_SENT_FEWSHOT_A = """
* "Inhalation is a process that relies on contraction of muscles while exhalation is a process that is usually—not always—passive because it relies on the relaxation of muscles."
* "Inhalation is an active process in which a person takes air into the body through the mouth and nose and pushes the air into the lungs."
* "The diaphragm and intercostal muscles contract during inhalation, while they relax during exhalation."
"""


async def pick_sent(doc, client):
    chat_prompt_with_inst = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=PICK_SENT_INSTRUCTION),
            HumanMessage(content=PICK_SENT_FEWSHOT_H),
            AIMessage(content=PICK_SENT_FEWSHOT_A),
            HumanMessagePromptTemplate.from_template("{user_message}"),
        ]
    )

    user_prompt_pos = {"user_message": f"Document: {doc}\n\nExtracted Key Sentences: "}
    chain = chat_prompt_with_inst | client
    result = await chain.ainvoke(user_prompt_pos)
    content = [
        e.replace("* ", "")
        for e in result.content.split("Here are the extracted key sentences:")[-1]
        .strip()
        .split("\n")
        if e.startswith("* ") and e.replace("* ", "") != ""
    ]
    return content

EXTRACT_SENTENCE_INSTRUCTION = """Given the following document, extract key sentences that contain possible relevant contexts. Focus on sentences that capture the main ideas, important details, or essential concepts relevant to the document's content."""
EXTRACT_SENTENCE_FEWSHOT_H = """Document: Breathing is an active process requiring the contraction of skeletal muscles. The primary muscles of respiration include the external intercostal muscles (located between the ribs) and the diaphragm (a sheet of muscle located between the thoracic & abdominal cavities). Breathing is a physical process consisting of inhalation, exhalation, and relaxation. Inhalation is an active process while exhalation is passive. Breathing involves two stages known as ventilation and gas exchange. Ventilation is the movement of the air in and out of the lungs. Exhalation—Carbon dioxide is pushed out of your body through the nose or mouth when your diaphragm contracts. At the site of the alveoli, the oxygen goes into the oxygen-poor blood that is coming from all over the body. The carbon dioxide is then forced out and breathed out. Answered by The Community. Making the world better, one answer at a time. Inhalation is a process that relies on contraction of muscles while exhalation is a process that is usually—not always—passive because it relies on the relaxation of muscles. However, when you speak, sing, or blow out a candle, the muscles between the ribs and abs contract, making it active. An active process is one that is carried out at the expense of energy, and a passive process occurs spontaneously in nature. For example, consider two solutions separated by a membrane; one side has a higher concentration of glucose than the other. • Inhalation is an active process, whereas exhalation is a passive process. • Exhalation occurs followed by inhalation. • The diaphragm and intercostal muscles contract during inhalation, while they relax during exhalation. Inhalation is an active process in which a person takes air into the body through the mouth and nose and pushes the air into the lungs. Inhalation is controlled by the brain. During the process of inhalation, the diaphragm and intercostal muscle contractions cause the thoracic cavity to enlarge. This creates a slight vacuum condition due to decreasing air pressure in the lungs. Due to the pressure gradient between the atmosphere and thoracic cavity, air moves into the lungs via the trachea.

Extracted Key Sentences: """
EXTRACT_SENTENCE_FEWSHOT_A = """
* "Inhalation is an active process while exhalation is passive."
* "Inhalation is a process that relies on contraction of muscles while exhalation is a process that is usually—not always—passive because it relies on the relaxation of muscles."
* "Inhalation is an active process, whereas exhalation is a passive process."
* "The diaphragm and intercostal muscles contract during inhalation, while they relax during exhalation."
* "Inhalation is an active process in which a person takes air into the body through the mouth and nose and pushes the air into the lungs."
"""


async def summary_extract_async_v2(doc, client):
    chat_prompt_with_inst = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=EXTRACT_SENTENCE_INSTRUCTION),
            HumanMessage(content=EXTRACT_SENTENCE_FEWSHOT_H),
            AIMessage(content=EXTRACT_SENTENCE_FEWSHOT_A),
            HumanMessagePromptTemplate.from_template("{user_message}"),
        ]
    )

    user_prompt_pos = {"user_message": f"Document: {doc}\n\nExtracted Key Sentences: "}
    chain = chat_prompt_with_inst | client
    result = await chain.ainvoke(user_prompt_pos)
    content = [
        e.replace("* ", "")
        for e in result.content.split("Here are the extracted key sentences:")[-1]
        .strip()
        .split("\n")
        if e.startswith("* ") and e.replace("* ", "") != ""
    ]
    return content


QGEN_FROM_KEY_SENTENCE_INSTRUCTION = """Given the document and the extracted key sentences, generate a relevant query that could be used to retrieve this document in an information retrieval system. The query should be concise and capture the main idea or question addressed by the document, incorporating important keywords and concepts from the key sentences. Output only the query text without any additional text or explanation."""
QGEN_FROM_KEY_SENTENCE_FEWSHOT_H = """Document: Breathing is an active process requiring the contraction of skeletal muscles. The primary muscles of respiration include the external intercostal muscles (located between the ribs) and the diaphragm (a sheet of muscle located between the thoracic & abdominal cavities). Breathing is a physical process consisting of inhalation, exhalation, and relaxation. Inhalation is an active process while exhalation is passive. Breathing involves two stages known as ventilation and gas exchange. Ventilation is the movement of the air in and out of the lungs. Exhalation—Carbon dioxide is pushed out of your body through the nose or mouth when your diaphragm contracts. At the site of the alveoli, the oxygen goes into the oxygen-poor blood that is coming from all over the body. The carbon dioxide is then forced out and breathed out. Answered by The Community. Making the world better, one answer at a time. Inhalation is a process that relies on contraction of muscles while exhalation is a process that is usually—not always—passive because it relies on the relaxation of muscles. However, when you speak, sing, or blow out a candle, the muscles between the ribs and abs contract, making it active. An active process is one that is carried out at the expense of energy, and a passive process occurs spontaneously in nature. For example, consider two solutions separated by a membrane; one side has a higher concentration of glucose than the other. • Inhalation is an active process, whereas exhalation is a passive process. • Exhalation occurs followed by inhalation. • The diaphragm and intercostal muscles contract during inhalation, while they relax during exhalation. Inhalation is an active process in which a person takes air into the body through the mouth and nose and pushes the air into the lungs. Inhalation is controlled by the brain. During the process of inhalation, the diaphragm and intercostal muscle contractions cause the thoracic cavity to enlarge. This creates a slight vacuum condition due to decreasing air pressure in the lungs. Due to the pressure gradient between the atmosphere and thoracic cavity, air moves into the lungs via the trachea.

Key Sentence: "Inhalation is an active process while exhalation is passive."

Generated Query: """
QGEN_FROM_KEY_SENTENCE_FEWSHOT_A = (
    "Why is inhalation considered an active process while exhalation is passive?"
)


async def duqgen_v2(doc, key, client):
    chat_prompt_with_inst = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=QGEN_FROM_KEY_SENTENCE_INSTRUCTION),
            HumanMessage(content=QGEN_FROM_KEY_SENTENCE_FEWSHOT_H),
            AIMessage(content=QGEN_FROM_KEY_SENTENCE_FEWSHOT_A),
            HumanMessagePromptTemplate.from_template("{user_message}"),
        ]
    )
    user_prompt_pos = {
        "user_message": f"Document: {doc}\n\nKey Sentence: {key}\n\nGenerated Query: "
    }
    chain = chat_prompt_with_inst | client
    result = await chain.ainvoke(user_prompt_pos)
    content = result.content.strip()
    return content


RAQGEN_INSTRUCTION = "I have a document and its retrieved neighbors. Based on the document and its neighboring contexts, generate a query relevant to the document."


async def raqgen(doc, nearest, client, only_count_tokens_input=False):
    chat_prompt_with_inst = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=QGEN_FROM_KEY_SENTENCE_INSTRUCTION),
            HumanMessagePromptTemplate.from_template("{user_message}"),
        ]
    )
    nearest_number_joined = "\n".join(
        [f"Neighbor {i+1}: {nearest[i]}" for i in range(len(nearest))]
    )
    user_prompt_args = {
        "user_message": f"Document: {doc}\n\nNeighboring Documents: \n{nearest_number_joined}\n\nGenerated Query: "
    }

    if only_count_tokens_input:
        full_prompt = chat_prompt_with_inst.format(**user_prompt_args)
        enc = tiktoken.encoding_for_model("gpt-4o")
        encoded = enc.encode(full_prompt)
        return len(encoded)

    chain = chat_prompt_with_inst | client
    result = await chain.ainvoke(user_prompt_args)
    content = result.content.strip()
    print("---------------------------\n")
    print("Content: ")
    print(content)
    print("---------------------------\n")
    return content


async def raqgen_batch(docs, nearests, client):
    chat_prompt_with_inst = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=QGEN_FROM_KEY_SENTENCE_INSTRUCTION),
            HumanMessagePromptTemplate.from_template("{user_message}"),
        ]
    )
    inputs = []
    for doc, nearest in zip(docs, nearests):
        nearest_number_joined = "\n".join(
            [f"Neighbor {i+1}: {nearest[i]}" for i in range(len(nearest))]
        )
        user_prompt_args = {
            "user_message": f"Document: {doc}\n\nNeighboring Documents: \n{nearest_number_joined}\n\nGenerated Query: "
        }
        inputs.append(user_prompt_args)

    chain = chat_prompt_with_inst | client
    results = await chain.ainvoke(inputs)
    return [result.content.strip() for result in results]
