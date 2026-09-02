import json
import re

query_raw = '''[1] What determines the key date for establishing residency after moving states for tax purposes?
[2] How does moving from one state to another affect your income tax if the states have different tax rates?
[3] What steps should you take to establish residency in a new state when moving from a high tax state to a low tax state?
[4] Why is the location where you work important if you move your home but not your job?
[5] How do Virginia, Maryland, and DC handle income tax based on residency?
[6] What happens to your income tax if you live in Delaware but work in Virginia?
[7] What is the importance of researching reciprocity between two states before moving?
[8] What constitutes Massachusetts gross income for nonresidents and part-year residents?
[9] How can Massachusetts residents and part-year residents get a credit for taxes paid to another jurisdiction?
[10] Are nonresidents allowed to claim the taxes paid to other jurisdiction credit on their Massachusetts tax return?
[11] For which types of taxes is the other jurisdiction credit not allowed in Massachusetts?
[12] How is the credit for taxes paid to another jurisdiction calculated in Massachusetts?
[13] What steps must be taken to claim the other jurisdiction credit in Massachusetts?
[14] If you move to New Hampshire but continue to work in Massachusetts, what are your tax obligations?
[15] How does moving across state lines impact your tax situation?
[16] What are the financial implications of establishing residency in a new state?
[17] How does the date of residency establishment affect state income tax allocation?
[18] What documentation is necessary to establish residency in a new state for tax purposes?
[19] How does the concept of tax reciprocity between states affect individuals who move?
[20] What is the tax treatment for part-year residents in Massachusetts?
[21] How does income sourced within Massachusetts affect nonresident tax filings?
[22] What are the exclusions for the other jurisdiction tax credit in Massachusetts?
[23] How do you determine eligibility for the other jurisdiction tax credit in Massachusetts?
[24] What is the process for claiming income taxes paid to other jurisdictions as a credit in Massachusetts?
[25] How do state tax differences influence the decision to move to a new state?
[26] What does establishing residency entail for tax purposes when moving to a low tax state?
[27] Why might someone choose to register their vehicle and vote in a new state after moving?
[28] What are the consequences of not establishing residency when moving to a new state for work?
[29] How can individuals navigate income tax laws when moving to a new state?
[30] What advice is given to those moving from one state to another regarding tax planning?
[31] How do Massachusetts tax laws apply to income earned in the state by nonresidents?
[32] Why is it crucial for part-year residents in Massachusetts to understand the tax credit for other jurisdictions?
[33] How does a change in residency affect income tax obligations in the original state?
[34] What impact does inter-state mobility have on personal finances due to tax laws?
[35] What role does the drivers’ license play in establishing new residency for tax purposes?
[36] How do reciprocal tax agreements between states influence where people decide to live and work?
[37] What are the tax implications for Massachusetts residents working in a different state?
[38] How can newly moved individuals ensure compliance with state tax laws in their new home state?
[39] What specific actions can lead to a clear establishment of new residency for someone who has recently moved?
[40] How does moving from a state with high taxes to one with lower taxes affect your annual tax returns?
[41] What special tax considerations exist for individuals moving within the DMV region?
[42] How do Massachusetts' tax laws regarding nonresident and part-year resident income differ from other states?
[43] What can lead to a requirement to pay income tax to two states?
[44] How does one formally change their state of residency for tax purposes after moving?
[45] What financial planning advice is recommended for those considering an interstate move?
[46] How does the taxation of income earned out of state work for Massachusetts residents?
[47] What steps are involved in calculating the credit for taxes paid to another jurisdiction for Massachusetts tax purposes?
[48] Are there any tax benefits for individuals moving to a state without income tax while working in a state with income tax?
[49] What challenges might arise from moving your home to a new state without changing your place of employment?
[50] How do local and state taxes interact for individuals living in one state and working in another?
[51] What documentation is considered critical when proving new residency for state tax purposes?
[52] How do the income tax rules for living in Delaware and working in Virginia reflect broader state tax policies?
[53] What strategies can individuals employ to minimize their tax liability when moving between states with different tax rates?
[54] Why is it necessary to understand the tax rules of both the old and new state when planning an interstate move?
[55] What role does voter registration play in the process of establishing residency for tax purposes?
[56] How does the concept of tax domicile affect interstate movers?
[57] What tax planning considerations should be made before moving to another state for work?
[58] How does commuting from one state to another for work impact your tax filings?
[59] What are the implications of part-year residency on state income tax filings?
[60] How can the other jurisdiction tax credit reduce the tax burden on Massachusetts taxpayers?
[61] What are the criteria for sourcing income to Massachusetts for tax purposes?
[62] How does dual-residency affect state income tax obligations?
[63] What are the implications of not correctly establishing residency after moving states?
[64] What is necessary for taxpayers to properly apply for the other jurisdiction credit on a Massachusetts return?
[65] How do state-specific tax laws impact individuals who move from one state with high taxes to one with lower taxes?
[66] Why is it important for individuals to update their driver's license after moving to a new state?
[67] How do the income tax requirements differ for someone moving from Massachusetts to New Hampshire?
[68] What problems might arise if you fail to establish residency correctly in a new state?
[69] How does residency affect state income tax liabilities for individuals?
[70] What are the financial benefits of understanding reciprocity agreements between states before moving?
[71] How can the date of establishing new residency impact fiscal obligations to different states?
[72] What specific actions can indicate the intent to establish residency in a new state?
[73] Why might an individual be required to pay taxes in two states after moving?
[74] How do you prove residency for tax purposes after relocating to a different state?
[75] What are the consequences of moving to a state with reciprocal tax agreements?
[76] How do local and state tax considerations influence someone’s decision to move to a new state?
[77] What are the tax considerations for individuals working remotely in a different state from their employer?
[78] Can maintaining employment in a high-tax state affect your residency status in a new state?
[79] How does the physical move of residency influence state income tax responsibilities?
[80] What is the importance of vehicle registration in the context of establishing new state residency for tax purposes?
[81] How can failure to properly establish new residency impact tax liabilities and penalties?
[82] What are the implications of residency rules for taxing authorities in different states?
[83] How do states handle taxation of individuals who live in one state but work in another?
[84] What tax planning steps are advised for individuals considering an inter-state move?
[85] What changes in tax obligations can one expect when moving from a high-tax state to a lower-tax state?
[86] Why is the physical act of moving not always sufficient to establish tax residency in a new state?
[87] What are the legal ramifications of misrepresenting residency for tax purposes?
[88] How do tax jurisdictions determine the source of income for tax purposes?
[89] Why is it significant to keep detailed records when moving states for tax purposes?
[90] What is the role of state income taxes in the decision-making process of moving?
[91] How do tax credits work for individuals who pay taxes to multiple states?
[92] Why might some states require tax payments from individuals who no longer reside there but continue to work in the state?
[93] What are the tax implications for individuals moving from a no-tax state to a state with income tax?
[94] How can interstate movers ensure they are paying the correct amount of state income tax?
[95] What legal documents should be updated promptly upon moving to a new state to ensure compliance with state tax laws?
[96] How does an individual's intent and actions in a new state affect their tax residency status?
[97] What are the key considerations for tax planning when moving to a state with different tax laws?
[98] How does a part-year residency status affect one’s state tax return?
[99] What specific measures can help someone prove their tax residency in a new state after moving?
[100] How can understanding state tax laws benefit someone planning to move across state lines?'''

query_split_regex = re.compile(r'\[\d+\]\s*([^\[\]]+)')
queries = query_split_regex.findall(query_raw)
for query in queries:
    query = ' '.join(query.split())
    dump = json.dumps({'did': '571430', 'query': query})
    print(dump)
