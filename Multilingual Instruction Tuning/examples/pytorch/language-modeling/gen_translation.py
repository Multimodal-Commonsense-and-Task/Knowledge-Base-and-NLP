# https://github.com/NJUNLP/x-LLM/blob/main/data/translation/translation.py
import json
import random
import shutil


en_inst = [
    "Translate the following sentences from {source_lang} to {target_lang}.",
    "What do the following sentences mean in {target_lang}?",
    "Please provide the {target_lang} translation for the following sentences.",
    "Convert the subsequent sentences from {source_lang} into {target_lang}.",
    "Render the listed sentences in {target_lang} from their original {source_lang} form.",
    "Transform the upcoming sentences from {source_lang} language to {target_lang} language.",
    "Change the given sentences from {source_lang} to {target_lang} format.",
    "Turn the following sentences from their {source_lang} version to the {target_lang} version.",
    "Adapt the mentioned sentences from {source_lang} to the {target_lang} language.",
    "Transpose the next sentences from the {source_lang} format to the {target_lang} format.",
    "Switch the specified sentences from their {source_lang} form to {target_lang} form.",
    "Reinterpret the ensuing sentences from {source_lang} to {target_lang} language.",
    "Modify the forthcoming sentences, converting them from {source_lang} to {target_lang}.",
    "How can the subsequent sentences be interpreted in {target_lang}?",
    "What is the meaning of these sentences when translated to {target_lang}?",
    "In the context of {target_lang}, what do the upcoming sentences signify?",
    "How would you express the meaning of the following sentences in {target_lang}?",
    "What is the significance of the mentioned sentences in {target_lang}?",
    "In {target_lang}, what do the given sentences convey?",
    "When translated to {target_lang}, what message do these sentences carry?",
    "What is the intended meaning of the ensuing sentences in {target_lang}?",
    "How should the following sentences be comprehended in {target_lang}?",
    "In terms of {target_lang}, what do the next sentences imply?",
    "Kindly furnish the {target_lang} translation of the subsequent sentences.",
    "Could you supply the {target_lang} translation for the upcoming sentences?",
    "Please offer the {target_lang} rendition for the following statements.",
    "I'd appreciate it if you could present the {target_lang} translation for these sentences.",
    "Can you deliver the {target_lang} translation for the mentioned sentences?",
    "Please share the {target_lang} version of the given sentences.",
    "It would be helpful if you could provide the {target_lang} translation of the ensuing sentences.",
    "Kindly submit the {target_lang} interpretation for the next sentences.",
    "Please make available the {target_lang} translation for the listed sentences.",
    "Can you reveal the {target_lang} translation of the forthcoming sentences?",
]

srcshortlang, srcflorescode, srclang3code, srclonglang = 'en', 'eng', 'eng', "English"
for shortlang, florescode, lang3code, longlang in [
    # ("ar", "ara", "arb", "Arabic"),
    # ("hi", "hin", "hin", "Hindi"),
    # ("sw", "swh", "swa", "Swahili"),
    # ("th", "tha", "tha", "Thai"),
    # ("tr", "tur", "tur", "Turkish"),
    # ("bn", "ben", "ben", "Bengali"),
    # ("ta", "tam", "tam", "Tamil"),
    # ("te", "tel", "tel", "Telugu"),
    # ("el", "ell", "ell", "Greek"),
    ("gu", "guj", "guj", "Gujarati"),
    ("kn", "kan", "kan", "Kannada"),
    ("ml", "mal", "mal", "Malayalam"),
    ("mr", "mar", "mar", "Marathi"),
    ("pa", "pan", "pan", "Panjabi"),
]:
    print(longlang)
    for inst in en_inst:
        print(f""""{inst.format_map({'source_lang': longlang, 'target_lang': srclonglang})}",""")


_INSTRUCTIONS = {
    "en" : [
    "Translate the following sentences from {source_lang} to {target_lang}.",
    "What do the following sentences mean in {target_lang}?",
    "Please provide the {target_lang} translation for the following sentences.",
    "Convert the subsequent sentences from {source_lang} into {target_lang}.",
    "Render the listed sentences in {target_lang} from their original {source_lang} form.",
    "Transform the upcoming sentences from {source_lang} language to {target_lang} language.",
    "Change the given sentences from {source_lang} to {target_lang} format.",
    "Turn the following sentences from their {source_lang} version to the {target_lang} version.",
    "Adapt the mentioned sentences from {source_lang} to the {target_lang} language.",
    "Transpose the next sentences from the {source_lang} format to the {target_lang} format.",
    "Switch the specified sentences from their {source_lang} form to {target_lang} form.",
    "Reinterpret the ensuing sentences from {source_lang} to {target_lang} language.",
    "Modify the forthcoming sentences, converting them from {source_lang} to {target_lang}.",
    "How can the subsequent sentences be interpreted in {target_lang}?",
    "What is the meaning of these sentences when translated to {target_lang}?",
    "In the context of {target_lang}, what do the upcoming sentences signify?",
    "How would you express the meaning of the following sentences in {target_lang}?",
    "What is the significance of the mentioned sentences in {target_lang}?",
    "In {target_lang}, what do the given sentences convey?",
    "When translated to {target_lang}, what message do these sentences carry?",
    "What is the intended meaning of the ensuing sentences in {target_lang}?",
    "How should the following sentences be comprehended in {target_lang}?",
    "In terms of {target_lang}, what do the next sentences imply?",
    "Kindly furnish the {target_lang} translation of the subsequent sentences.",
    "Could you supply the {target_lang} translation for the upcoming sentences?",
    "Please offer the {target_lang} rendition for the following statements.",
    "I'd appreciate it if you could present the {target_lang} translation for these sentences.",
    "Can you deliver the {target_lang} translation for the mentioned sentences?",
    "Please share the {target_lang} version of the given sentences.",
    "It would be helpful if you could provide the {target_lang} translation of the ensuing sentences.",
    "Kindly submit the {target_lang} interpretation for the next sentences.",
    "Please make available the {target_lang} translation for the listed sentences.",
    "Can you reveal the {target_lang} translation of the forthcoming sentences?",
],

    "bn" : [
"নিম্নলিখিত বাক্যগুলো বাংলা থেকে ইংরেজিতে অনুবাদ করুন।",
"ইংরেজিতে নিচের বাক্যগুলোর মানে কি?",
"নিম্নলিখিত বাক্যের ইংরেজি অনুবাদ প্রদান করুন।",
"পরবর্তী বাক্যকে বাংলা থেকে ইংরেজিতে রূপান্তর করুন।",
"তাদের মূল বাংলা ফর্ম থেকে ইংরেজিতে তালিকাভুক্ত বাক্য রেন্ডার করুন।",
"আসন্ন বাক্যকে বাংলা ভাষা থেকে ইংরেজি ভাষায় রূপান্তর করুন।",
"প্রদত্ত বাক্যগুলিকে বাংলা থেকে ইংরেজি বিন্যাসে পরিবর্তন করুন।",
"নিম্নলিখিত বাক্যগুলোকে তাদের বাংলা সংস্করণ থেকে ইংরেজি সংস্করণে পরিণত করুন।",
"উল্লেখিত বাক্যগুলিকে বাংলা থেকে ইংরেজি ভাষায় অভিযোজিত করুন।",
"বাংলা ফরম্যাট থেকে পরবর্তী বাক্যগুলো ইংরেজি বিন্যাসে স্থানান্তর করুন।",
"নির্দিষ্ট বাক্যগুলিকে তাদের বাংলা ফর্ম থেকে ইংরেজি ফর্মে পরিবর্তন করুন।",
"আগামী বাক্যগুলিকে বাংলা থেকে ইংরেজি ভাষায় পুনঃব্যাখ্যা করুন।",
"আগামী বাক্যগুলিকে বাংলা থেকে ইংরেজিতে রূপান্তর করে পরিবর্তন করুন।",
"কিভাবে পরবর্তী বাক্যগুলি ইংরেজিতে ব্যাখ্যা করা যায়?",
"ইংরেজিতে অনুবাদ করা হলে এই বাক্যগুলির অর্থ কী?",
"ইংরেজি প্রসঙ্গে, আসন্ন বাক্যগুলি কী বোঝায়?",
"আপনি কীভাবে ইংরেজিতে নিম্নলিখিত বাক্যগুলির অর্থ প্রকাশ করবেন?",
"ইংরেজিতে উল্লেখিত বাক্যগুলোর তাৎপর্য কি?",
"ইংরেজিতে, প্রদত্ত বাক্যগুলি কী বোঝায়?",
"ইংরেজিতে অনুবাদ করা হলে, এই বাক্যগুলো কী বার্তা বহন করে?",
"ইংরেজিতে পরবর্তী বাক্যগুলির উদ্দেশ্যমূলক অর্থ কী?",
"ইংরেজিতে নিম্নলিখিত বাক্যগুলি কীভাবে বোঝা উচিত?",
"ইংরেজির পরিপ্রেক্ষিতে, পরবর্তী বাক্যগুলি কী বোঝায়?",
"দয়া করে পরবর্তী বাক্যগুলোর ইংরেজি অনুবাদ প্রদান করুন।",
"আপনি কি আসন্ন বাক্যগুলির জন্য ইংরেজি অনুবাদ সরবরাহ করতে পারেন?",
"নিম্নলিখিত বিবৃতিগুলির জন্য অনুগ্রহ করে ইংরেজি উপস্থাপনা অফার করুন।",
"আপনি যদি এই বাক্যগুলির ইংরেজি অনুবাদ উপস্থাপন করতে পারেন তবে আমি এটির প্রশংসা করব।",
"আপনি কি উল্লিখিত বাক্যের ইংরেজি অনুবাদ দিতে পারবেন?",
"দয়া করে প্রদত্ত বাক্যের ইংরেজি সংস্করণ শেয়ার করুন।",
"আপনি যদি পরবর্তী বাক্যগুলির ইংরেজি অনুবাদ প্রদান করতে পারেন তবে এটি সহায়ক হবে।",
"দয়া করে পরবর্তী বাক্যগুলির জন্য ইংরেজি ব্যাখ্যা জমা দিন।",
"অনুগ্রহ করে তালিকাভুক্ত বাক্যের জন্য ইংরেজি অনুবাদ উপলব্ধ করুন।",
"আপনি কি আসন্ন বাক্যের ইংরেজি অনুবাদ প্রকাশ করতে পারেন?",

    ],

    "ta": [
"பின்வரும் வாக்கியங்களை தமிழிலிருந்து ஆங்கிலத்திற்கு மொழிபெயர்க்கவும்.",
"பின்வரும் வாக்கியங்கள் ஆங்கிலத்தில் என்ன அர்த்தம்?",
"பின்வரும் வாக்கியங்களுக்கு ஆங்கில மொழிபெயர்ப்பை வழங்கவும்.",
"அடுத்தடுத்த வாக்கியங்களை தமிழில் இருந்து ஆங்கிலத்திற்கு மாற்றவும்.",
"பட்டியலிடப்பட்ட வாக்கியங்களை அவற்றின் அசல் தமிழ் வடிவத்திலிருந்து ஆங்கிலத்தில் வழங்கவும்.",
"வரவிருக்கும் வாக்கியங்களை தமிழ் மொழியிலிருந்து ஆங்கில மொழிக்கு மாற்றவும்.",
"கொடுக்கப்பட்ட வாக்கியங்களை தமிழில் இருந்து ஆங்கில வடிவத்திற்கு மாற்றவும்.",
"பின்வரும் வாக்கியங்களை அவற்றின் தமிழ்ப் பதிப்பிலிருந்து ஆங்கிலப் பதிப்பிற்கு மாற்றவும்.",
"குறிப்பிடப்பட்ட வாக்கியங்களை தமிழில் இருந்து ஆங்கில மொழிக்கு மாற்றியமைக்கவும்.",
"அடுத்த வாக்கியங்களை தமிழ் வடிவத்திலிருந்து ஆங்கில வடிவத்திற்கு மாற்றவும்.",
"குறிப்பிட்ட வாக்கியங்களை அவற்றின் தமிழ் வடிவத்திலிருந்து ஆங்கில வடிவத்திற்கு மாற்றவும்.",
"தமிழில் இருந்து ஆங்கிலத்திற்கு வரும் வாக்கியங்களை மறுவிளக்கம் செய்யுங்கள்.",
"வரவிருக்கும் வாக்கியங்களைத் தமிழிலிருந்து ஆங்கிலத்திற்கு மாற்றுதல்.",
"அடுத்தடுத்த வாக்கியங்களை ஆங்கிலத்தில் எப்படி விளக்குவது?",
"ஆங்கிலத்தில் மொழிபெயர்க்கப்பட்ட இந்த வாக்கியங்களின் அர்த்தம் என்ன?",
"ஆங்கிலத்தின் சூழலில், வரவிருக்கும் வாக்கியங்கள் எதைக் குறிக்கின்றன?",
"பின்வரும் வாக்கியங்களின் அர்த்தத்தை ஆங்கிலத்தில் எப்படி வெளிப்படுத்துவீர்கள்?",
"ஆங்கிலத்தில் குறிப்பிடப்பட்டுள்ள வாக்கியங்களின் முக்கியத்துவம் என்ன?",
"ஆங்கிலத்தில், கொடுக்கப்பட்ட வாக்கியங்கள் எதை உணர்த்துகின்றன?",
"ஆங்கிலத்தில் மொழிபெயர்க்கும்போது, இந்த வாக்கியங்கள் என்ன செய்தியைக் கொண்டு செல்கின்றன?",
"ஆங்கிலத்தில் வரும் வாக்கியங்களின் நோக்கம் என்ன?",
"பின்வரும் வாக்கியங்களை ஆங்கிலத்தில் எவ்வாறு புரிந்து கொள்ள வேண்டும்?",
"ஆங்கிலத்தைப் பொறுத்தவரை, அடுத்த வாக்கியங்கள் எதைக் குறிக்கின்றன?",
"பின்வரும் வாக்கியங்களின் ஆங்கில மொழிபெயர்ப்பை தயவு செய்து தரவும்.",
"வரவிருக்கும் வாக்கியங்களுக்கு ஆங்கில மொழிபெயர்ப்பை வழங்க முடியுமா?",
"பின்வரும் அறிக்கைகளுக்கு ஆங்கில விளக்கத்தை வழங்கவும்.",
"இந்த வாக்கியங்களுக்கான ஆங்கில மொழிபெயர்ப்பை நீங்கள் வழங்கினால் நான் மிகவும் பாராட்டுகிறேன்.",
"குறிப்பிடப்பட்ட வாக்கியங்களுக்கான ஆங்கில மொழிபெயர்ப்பை வழங்க முடியுமா?",
"தயவுசெய்து கொடுக்கப்பட்ட வாக்கியங்களின் ஆங்கிலப் பதிப்பைப் பகிரவும்.",
"தொடர்ந்து வரும் வாக்கியங்களின் ஆங்கில மொழிபெயர்ப்பை நீங்கள் வழங்கினால் உதவியாக இருக்கும்.",
"அடுத்த வாக்கியங்களுக்கான ஆங்கில விளக்கத்தை சமர்ப்பிக்கவும்.",
"பட்டியலிடப்பட்ட வாக்கியங்களுக்கான ஆங்கில மொழிபெயர்ப்பைக் கிடைக்கச் செய்யவும்.",
"வரவிருக்கும் வாக்கியங்களின் ஆங்கில மொழிபெயர்ப்பை வெளிப்படுத்த முடியுமா?",


    ],

    "te" : [
"క్రింది వాక్యాలను తెలుగు నుండి ఆంగ్లంలోకి అనువదించండి.",
"ఇంగ్లీషులో కింది వాక్యాల అర్థం ఏమిటి?",
"దయచేసి కింది వాక్యాలకు ఆంగ్ల అనువాదాన్ని అందించండి.",
"తర్వాత వాక్యాలను తెలుగు నుండి ఆంగ్లంలోకి మార్చండి.",
"లిస్టెడ్ వాక్యాలను వాటి అసలు తెలుగు రూపం నుండి ఆంగ్లంలో రెండర్ చేయండి.",
"రాబోయే వాక్యాలను తెలుగు భాష నుండి ఆంగ్ల భాషలోకి మార్చండి.",
"ఇచ్చిన వాక్యాలను తెలుగు నుండి ఆంగ్ల ఆకృతికి మార్చండి.",
"క్రింది వాక్యాలను వాటి తెలుగు వెర్షన్ నుండి ఆంగ్ల వెర్షన్‌కి మార్చండి.",
"పేర్కొన్న వాక్యాలను తెలుగు నుండి ఆంగ్ల భాషకు మార్చండి.",
"తదుపరి వాక్యాలను తెలుగు ఫార్మాట్ నుండి ఆంగ్ల ఆకృతికి మార్చండి.",
"పేర్కొన్న వాక్యాలను వాటి తెలుగు రూపం నుండి ఆంగ్ల రూపానికి మార్చండి.",
"తదుపరి వాక్యాలను తెలుగు నుండి ఆంగ్ల భాషలోకి తిరిగి అర్థం చేసుకోండి.",
"రాబోయే వాక్యాలను సవరించండి, వాటిని తెలుగు నుండి ఆంగ్లంలోకి మార్చండి.",
"తరువాతి వాక్యాలను ఆంగ్లంలో ఎలా అన్వయించవచ్చు?",
"ఇంగ్లీషులోకి అనువదించబడినప్పుడు ఈ వాక్యాల అర్థం ఏమిటి?",
"ఇంగ్లీష్ సందర్భంలో, రాబోయే వాక్యాలు దేనిని సూచిస్తాయి?",
"మీరు క్రింది వాక్యాల అర్థాన్ని ఆంగ్లంలో ఎలా వ్యక్తపరుస్తారు?",
"ఇంగ్లీషులో పేర్కొన్న వాక్యాల ప్రాముఖ్యత ఏమిటి?",
"ఇంగ్లీషులో, ఇచ్చిన వాక్యాలు ఏమి తెలియజేస్తాయి?",
"ఇంగ్లీషులోకి అనువదించబడినప్పుడు, ఈ వాక్యాలు ఏ సందేశాన్ని అందిస్తాయి?",
"ఇంగ్లీష్‌లో తదుపరి వాక్యాలకు ఉద్దేశించిన అర్థం ఏమిటి?",
"క్రింది వాక్యాలను ఆంగ్లంలో ఎలా గ్రహించాలి?",
"ఇంగ్లీష్ పరంగా, తదుపరి వాక్యాలు ఏమి సూచిస్తాయి?",
"దయచేసి తదుపరి వాక్యాల ఆంగ్ల అనువాదాన్ని అందించండి.",
"రాబోయే వాక్యాల కోసం మీరు ఆంగ్ల అనువాదాన్ని అందించగలరా?",
"దయచేసి కింది స్టేట్‌మెంట్‌ల కోసం ఇంగ్లీష్ రెండిషన్‌ను అందించండి.",
"మీరు ఈ వాక్యాలకు ఆంగ్ల అనువాదాన్ని అందించగలిగితే నేను అభినందిస్తున్నాను.",
"పేర్కొన్న వాక్యాల కోసం మీరు ఆంగ్ల అనువాదాన్ని అందించగలరా?",
"దయచేసి ఇచ్చిన వాక్యాల ఆంగ్ల వెర్షన్‌ను భాగస్వామ్యం చేయండి.",
"మీరు తదుపరి వాక్యాల ఆంగ్ల అనువాదాన్ని అందించగలిగితే అది ఉపయోగకరంగా ఉంటుంది.",
"తరువాతి వాక్యాల కోసం దయచేసి ఆంగ్ల వివరణను సమర్పించండి.",
"దయచేసి జాబితా చేయబడిన వాక్యాల కోసం ఆంగ్ల అనువాదాన్ని అందుబాటులో ఉంచండి.",
"రాబోయే వాక్యాల ఆంగ్ల అనువాదాన్ని మీరు వెల్లడించగలరా?",
    ],

    "sw": [
"Tafsiri sentensi zifuatazo kutoka Kiswahili hadi Kiingereza.",
"Sentensi zifuatazo zinamaanisha nini kwa Kiingereza?",
"Tafadhali toa tafsiri ya Kiingereza kwa sentensi zifuatazo.",
"Badilisha sentensi zinazofuata kutoka Kiswahili hadi Kiingereza.",
"Toa sentensi zilizoorodheshwa kwa Kiingereza kutoka kwa umbo lao la asili la Kiswahili.",
"Badilisha sentensi zijazo kutoka lugha ya Kiswahili hadi lugha ya Kiingereza.",
"Badilisha sentensi ulizopewa kutoka kwa muundo wa Kiswahili hadi Kiingereza.",
"Geuza sentensi zifuatazo kutoka toleo lao la Kiswahili hadi toleo la Kiingereza.",
"Badilisha sentensi zilizotajwa kutoka kwa Kiswahili hadi lugha ya Kiingereza.",
"Tumia sentensi zinazofuata kutoka umbizo la Kiswahili hadi umbizo la Kiingereza.",
"Badilisha sentensi zilizoainishwa kutoka umbo lao la Kiswahili hadi umbo la Kiingereza.",
"Tafasiri upya sentensi zinazofuata kutoka lugha ya Kiswahili hadi Kiingereza.",
"Rekebisha sentensi zinazokuja, uzibadilishe kutoka Kiswahili hadi Kiingereza.",
"Je! Sentensi zinazofuata zinaweza kufasiriwaje kwa Kiingereza?",
"Ni nini maana ya sentensi hizi zinapotafsiriwa kwa Kiingereza?",
"Katika muktadha wa Kiingereza, sentensi zijazo zinamaanisha nini?",
"Unawezaje kueleza maana ya sentensi zifuatazo kwa Kiingereza?",
"Je, ni nini umuhimu wa sentensi zilizotajwa katika Kiingereza?",
"Kwa Kiingereza, sentensi zilizotolewa zinaonyesha nini?",
"Je, zinapotafsiriwa kwa Kiingereza, sentensi hizi huwa na ujumbe gani?",
"Ni nini maana iliyokusudiwa ya sentensi zinazofuata katika Kiingereza?",
"Je! Sentensi zifuatazo zinapaswa kueleweka vipi kwa Kiingereza?",
"Kwa upande wa Kiingereza, sentensi zifuatazo zinamaanisha nini?",
"Tafadhali toa tafsiri ya Kiingereza ya sentensi zinazofuata.",
"Je, unaweza kutoa tafsiri ya Kiingereza kwa sentensi zijazo?",
"Tafadhali toa toleo la Kiingereza kwa taarifa zifuatazo.",
"Ningeshukuru ikiwa unaweza kuwasilisha tafsiri ya Kiingereza kwa sentensi hizi.",
"Je, unaweza kutoa tafsiri ya Kiingereza kwa sentensi zilizotajwa?",
"Tafadhali shiriki toleo la Kiingereza la sentensi ulizopewa.",
"Ingesaidia ikiwa unaweza kutoa tafsiri ya Kiingereza ya sentensi zinazofuata.",
"Tafadhali wasilisha tafsiri ya Kiingereza kwa sentensi zinazofuata.",
"Tafadhali fanya kupatikana kwa tafsiri ya Kiingereza kwa sentensi zilizoorodheshwa.",
"Je, unaweza kufichua tafsiri ya Kiingereza ya sentensi zinazokuja?",
    ],
"hi": [
"निम्नलिखित वाक्यों का हिंदी से अंग्रेजी में अनुवाद करें।",
"निम्नलिखित वाक्यों का अंग्रेजी में क्या अर्थ है?",
"कृपया निम्नलिखित वाक्यों का अंग्रेजी अनुवाद प्रदान करें।"
"आगे के वाक्यों को हिंदी से अंग्रेजी में बदलें।",
"सूचीबद्ध वाक्यों को अंग्रेजी में उनके मूल हिंदी रूप से प्रस्तुत करें।",
"आने वाले वाक्यों को हिंदी भाषा से अंग्रेजी भाषा में बदलें।",
"दिए गए वाक्यों को हिंदी से अंग्रेजी प्रारूप में बदलें।",
"निम्नलिखित वाक्यों को उनके हिंदी संस्करण से अंग्रेजी संस्करण में बदलें।",
"उल्लेखित वाक्यों को हिंदी से अंग्रेजी भाषा में अपनाएं।",
"अगले वाक्यों को हिंदी प्रारूप से अंग्रेजी प्रारूप में बदलें।",
"निर्दिष्ट वाक्यों को उनके हिंदी रूप से अंग्रेजी रूप में बदलें।",
"आने वाले वाक्यों की हिंदी से अंग्रेजी भाषा में पुनर्व्याख्या करें।",
"आने वाले वाक्यों को हिंदी से अंग्रेजी में परिवर्तित करके संशोधित करें।"
"आगे के वाक्यों की अंग्रेजी में व्याख्या कैसे की जा सकती है?",
"अंग्रेजी में अनुवाद करने पर इन वाक्यों का क्या अर्थ है?",
"अंग्रेजी के संदर्भ में, आगामी वाक्य क्या दर्शाते हैं?",
"आप निम्नलिखित वाक्यों का अर्थ अंग्रेजी में कैसे व्यक्त करेंगे?",
"उल्लेखित वाक्यों का अंग्रेजी में क्या महत्व है?",
"अंग्रेजी में, दिए गए वाक्य क्या बताते हैं?",
"जब अंग्रेजी में अनुवाद किया जाता है, तो ये वाक्य क्या संदेश देते हैं?",
"अंग्रेजी में आगामी वाक्यों का अभिप्राय क्या है?",
"निम्नलिखित वाक्यों को अंग्रेजी में कैसे समझा जाना चाहिए?",
"अंग्रेजी के संदर्भ में, अगले वाक्यों का क्या अर्थ है?",
"कृपया अगले वाक्यों का अंग्रेजी अनुवाद प्रस्तुत करें।"
"क्या आप आगामी वाक्यों का अंग्रेजी अनुवाद उपलब्ध करा सकते हैं?"
"कृपया निम्नलिखित कथनों का अंग्रेजी अनुवाद प्रस्तुत करें।"
"यदि आप इन वाक्यों का अंग्रेजी अनुवाद प्रस्तुत कर सकें तो मुझे खुशी होगी।"
"क्या आप उल्लिखित वाक्यों का अंग्रेजी अनुवाद दे सकते हैं?"
"कृपया दिए गए वाक्यों का अंग्रेजी संस्करण साझा करें।"
"यदि आप आगामी वाक्यों का अंग्रेजी अनुवाद प्रदान कर सकें तो यह सहायक होगा।"
"कृपया अगले वाक्यों के लिए अंग्रेजी व्याख्या प्रस्तुत करें।"
"कृपया सूचीबद्ध वाक्यों का अंग्रेजी अनुवाद उपलब्ध कराएं।"
"क्या आप आगामी वाक्यों का अंग्रेजी अनुवाद बता सकते हैं?"
],
"th": [
"แปลประโยคต่อไปนี้จากไทยเป็นอังกฤษ",
"ประโยคต่อไปนี้หมายถึงอะไรในภาษาอังกฤษ?",
"โปรดระบุคำแปลภาษาอังกฤษสำหรับประโยคต่อไปนี้",
"แปลงประโยคต่อจากภาษาไทยเป็นภาษาอังกฤษ",
"แปลประโยคที่แสดงเป็นภาษาอังกฤษจากรูปภาษาไทยดั้งเดิม",
"แปลงประโยคต่อจากภาษาไทยเป็นภาษาอังกฤษ",
"เปลี่ยนประโยคที่กำหนดจากรูปแบบภาษาไทยเป็นภาษาอังกฤษ",
"เปลี่ยนประโยคต่อไปนี้จากฉบับภาษาไทยเป็นฉบับภาษาอังกฤษ",
"ดัดแปลงประโยคดังกล่าวจากภาษาไทยเป็นภาษาอังกฤษ",
"ย้ายประโยคถัดไปจากรูปแบบภาษาไทยเป็นรูปแบบภาษาอังกฤษ",
"เปลี่ยนประโยคที่ระบุจากแบบฟอร์มภาษาไทยเป็นภาษาอังกฤษ",
"ตีความประโยคที่ตามมาจากภาษาไทยเป็นภาษาอังกฤษใหม่",
"แก้ไขประโยคที่กำลังจะมีขึ้น แปลงจากภาษาไทยเป็นภาษาอังกฤษ",
"ประโยคต่อมาจะตีความเป็นภาษาอังกฤษได้อย่างไร",
"ประโยคเหล่านี้เมื่อแปลเป็นภาษาอังกฤษหมายความว่าอย่างไร",
"ในบริบทของภาษาอังกฤษ ประโยคต่อๆ ไปมีความหมายว่าอย่างไร",
"คุณจะอธิบายความหมายของประโยคต่อไปนี้เป็นภาษาอังกฤษได้อย่างไร",
"ประโยคที่กล่าวถึงในภาษาอังกฤษมีความสำคัญอย่างไร",
"ในภาษาอังกฤษ ประโยคที่ให้มาหมายถึงอะไร",
"เมื่อแปลเป็นภาษาอังกฤษ ประโยคเหล่านี้มีข้อความอะไรบ้าง",
"ความหมายที่ตั้งใจไว้ของประโยคที่ตามมาในภาษาอังกฤษคืออะไร",
"ประโยคต่อไปนี้ควรเข้าใจเป็นภาษาอังกฤษอย่างไร",
"ในแง่ของภาษาอังกฤษ ประโยคถัดไปหมายถึงอะไร",
"กรุณาจัดเตรียมคำแปลภาษาอังกฤษของประโยคต่อๆ ไป",
"คุณช่วยจัดหาคำแปลภาษาอังกฤษสำหรับประโยคที่กำลังจะมาถึงได้ไหม",
"โปรดเสนอเวอร์ชันภาษาอังกฤษสำหรับข้อความต่อไปนี้",
"ฉันจะขอบคุณมากหากคุณสามารถนำเสนอคำแปลภาษาอังกฤษสำหรับประโยคเหล่านี้",
"คุณช่วยส่งคำแปลภาษาอังกฤษสำหรับประโยคดังกล่าวได้ไหม",
"โปรดแบ่งปันประโยคที่กำหนดในเวอร์ชันภาษาอังกฤษ",
"จะเป็นประโยชน์มากหากคุณสามารถจัดเตรียมคำแปลภาษาอังกฤษของประโยคที่ตามมา",
"กรุณาส่งการตีความภาษาอังกฤษสำหรับประโยคถัดไป",
"โปรดจัดให้มีคำแปลภาษาอังกฤษสำหรับประโยคที่ระบุไว้",
"คุณช่วยเปิดเผยคำแปลภาษาอังกฤษของประโยคที่กำลังจะมาถึงได้ไหม",
],
"tr": [
"Aşağıdaki cümleleri Türkçeden İngilizceye çevirin.",
"Aşağıdaki cümleler İngilizce'de ne anlama geliyor?",
"Lütfen aşağıdaki cümlelerin İngilizce çevirisini sağlayın.",
"Sonraki cümleleri Türkçeden İngilizceye dönüştürün.",
"Listelenen cümleleri orijinal Türkçe formlarından İngilizce olarak aktarın.",
"Gelecek cümleleri Türkçeden İngilizceye dönüştürün.",
"Verilen cümleleri Türkçeden İngilizceye çevirin.",
"Aşağıdaki cümleleri Türkçe'den İngilizce'ye çevirin.",
"Belirtilen cümleleri Türkçeden İngilizceye uyarlayın.",
"Sonraki cümleleri Türkçe formatından İngilizce formatına aktarın.",
"Belirtilen cümleleri Türkçe biçiminden İngilizce biçimine çevirin.",
"Sonraki cümleleri Türkçeden İngilizceye yeniden yorumlayınız.",
"Gelecek cümleleri Türkçeden İngilizceye çevirerek değiştirin.",
"Sonraki cümleler İngilizce olarak nasıl yorumlanabilir?",
"Bu cümleler İngilizceye çevrildiğinde ne anlama geliyor?",
"İngilizce bağlamında gelecek cümleler ne anlama geliyor?",
"Aşağıdaki cümlelerin anlamını İngilizce olarak nasıl ifade edersiniz?",
"Bahsedilen cümlelerin İngilizce'deki önemi nedir?",
"İngilizce'de verilen cümleler ne ifade ediyor?",
"Bu cümleler İngilizceye çevrildiğinde nasıl bir mesaj taşıyor?",
"Sonraki cümlelerin İngilizce'deki anlamı nedir?",
"Aşağıdaki cümleler İngilizcede nasıl anlaşılmalıdır?",
"İngilizce açısından sonraki cümleler ne anlama geliyor?",
"Lütfen sonraki cümlelerin İngilizce çevirisini sağlayın.",
"Gelecek cümlelerin İngilizce çevirisini sağlayabilir misiniz?",
"Lütfen aşağıdaki ifadelerin İngilizce çevirisini sunun.",
"Bu cümlelerin İngilizce çevirisini sunarsanız çok sevinirim.",
"Bahsedilen cümlelerin İngilizce tercümesini gönderebilir misiniz?",
"Lütfen verilen cümlelerin İngilizce versiyonunu paylaşın.",
"Sonraki cümlelerin İngilizce çevirisini sağlayabilirseniz faydalı olur.",
"Lütfen sonraki cümlelerin İngilizce tercümesini gönderin.",
"Lütfen listelenen cümlelerin İngilizce çevirisini sağlayın.",
"Gelecek cümlelerin İngilizce çevirisini ortaya çıkarabilir misiniz?",
],
"ar": [
"ترجمة الجمل التالية من العربية إلى الإنجليزية.",
"ماذا تعني الجمل التالية باللغة الإنجليزية؟",
"يرجى تقديم الترجمة الإنجليزية للجمل التالية.",
"تحويل الجمل اللاحقة من العربية إلى الإنجليزية.",
"أخرج الجمل المدرجة باللغة الإنجليزية من صيغتها العربية الأصلية.",
"تحويل الجمل القادمة من اللغة العربية إلى اللغة الإنجليزية.",
"تغيير الجمل المعطاة من التنسيق العربي إلى الإنجليزي.",
"تحويل الجمل التالية من نسختها العربية إلى النسخة الإنجليزية.",
"تعديل الجمل المذكورة من اللغة العربية إلى اللغة الإنجليزية.",
"نقل الجمل التالية من التنسيق العربي إلى التنسيق الإنجليزي.",
"تحويل الجمل المحددة من شكلها العربي إلى شكلها الإنجليزي.",
"أعد تفسير الجمل التالية من اللغة العربية إلى اللغة الإنجليزية.",
"تعديل الجمل القادمة وتحويلها من العربية إلى الإنجليزية.",
"كيف يمكن تفسير الجمل اللاحقة باللغة الإنجليزية؟",
"ما معنى هذه الجمل عند ترجمتها إلى اللغة الإنجليزية؟",
"في سياق اللغة الإنجليزية، ماذا تعني الجمل القادمة؟",
"كيف تعبر عن معنى الجمل التالية باللغة الإنجليزية؟",
"ما أهمية الجمل المذكورة باللغة الإنجليزية؟",
"في اللغة الإنجليزية، ماذا تنقل الجمل المعطاة؟",
"عند ترجمتها إلى اللغة الإنجليزية، ما هي الرسالة التي تحملها هذه الجمل؟",
"ما هو المعنى المقصود من الجمل التالية باللغة الإنجليزية؟",
"كيف ينبغي فهم الجمل التالية باللغة الإنجليزية؟",
"فيما يتعلق باللغة الإنجليزية، ماذا تعني الجمل التالية؟",
"يرجى تقديم الترجمة الإنجليزية للجمل اللاحقة.",
"هل يمكنك توفير الترجمة الإنجليزية للجمل القادمة؟",
"يُرجى تقديم الترجمة الإنجليزية للعبارات التالية.",
"سأكون ممتنًا إذا أمكنك تقديم الترجمة الإنجليزية لهذه الجمل.",
"هل يمكنك تسليم الترجمة الإنجليزية للجمل المذكورة؟",
"يرجى مشاركة النسخة الإنجليزية من الجمل المقدمة.",
"سيكون من المفيد لو أمكنك توفير الترجمة الإنجليزية للجمل التالية.",
"يُرجى تقديم الترجمة الإنجليزية للجمل التالية.",
"يرجى توفير الترجمة الإنجليزية للجمل المذكورة.",
"هل يمكنك الكشف عن الترجمة الإنجليزية للجمل القادمة؟",
],
    'el': [
"Μεταφράστε τις παρακάτω προτάσεις από τα ελληνικά στα αγγλικά.",
"Τι σημαίνουν οι παρακάτω προτάσεις στα Αγγλικά;",
"Παρακαλώ παρέχετε την αγγλική μετάφραση για τις ακόλουθες προτάσεις.",
"Μετατρέψτε τις επόμενες προτάσεις από τα ελληνικά σε αγγλικά.",
"Αποδώστε τις προτάσεις που παρατίθενται στα αγγλικά από την αρχική τους ελληνική μορφή.",
"Μετατρέψτε τις επερχόμενες προτάσεις από την ελληνική γλώσσα στην αγγλική.",
"Αλλαγή των προτάσεων από ελληνική σε αγγλική μορφή.",
"Μετατρέψτε τις παρακάτω προτάσεις από την ελληνική τους έκδοση στην αγγλική.",
"Προσαρμόστε τις αναφερόμενες προτάσεις από την ελληνική στην αγγλική γλώσσα.",
"Μεταφέρετε τις επόμενες προτάσεις από την ελληνική μορφή στην αγγλική.",
"Αλλαγή των καθορισμένων προτάσεων από την ελληνική τους μορφή στην αγγλική.",
"Επανερμηνεύστε τις προτάσεις που ακολουθούν από την ελληνική στην αγγλική γλώσσα.",
"Τροποποιήστε τις προσεχείς προτάσεις, μετατρέποντάς τες από ελληνικά σε αγγλικά.",
"Πώς μπορούν οι επόμενες προτάσεις να ερμηνευθούν στα αγγλικά;",
"Ποιο είναι το νόημα αυτών των προτάσεων όταν μεταφράζονται στα αγγλικά;",
"Στο πλαίσιο των αγγλικών, τι σημαίνουν οι επερχόμενες προτάσεις;",
"Πώς θα εκφράσατε τη σημασία των παρακάτω προτάσεων στα Αγγλικά;",
"Ποια είναι η σημασία των αναφερόμενων προτάσεων στα αγγλικά;",
"Στα αγγλικά, τι μεταδίδουν οι προτάσεις;",
"Όταν μεταφράζονται στα αγγλικά, τι μήνυμα φέρουν αυτές οι προτάσεις;",
"Ποιο είναι το επιδιωκόμενο νόημα των προτάσεων που ακολουθούν στα αγγλικά;",
"Πώς πρέπει να κατανοηθούν οι ακόλουθες προτάσεις στα αγγλικά;",
"Όσον αφορά τα αγγλικά, τι υποδηλώνουν οι επόμενες προτάσεις;",
"Παρακαλούμε δώστε την αγγλική μετάφραση των επόμενων προτάσεων.",
"Θα μπορούσατε να παράσχετε την αγγλική μετάφραση για τις επερχόμενες προτάσεις;",
"Παρακαλώ προσφέρετε την αγγλική απόδοση για τις ακόλουθες δηλώσεις.",
"Θα το εκτιμούσα αν μπορούσατε να παρουσιάσετε την αγγλική μετάφραση για αυτές τις προτάσεις.",
"Μπορείτε να παραδώσετε την αγγλική μετάφραση για τις αναφερόμενες προτάσεις;",
"Παρακαλώ κοινοποιήστε την αγγλική έκδοση των προτάσεων που δίνονται.",
"Θα ήταν χρήσιμο αν μπορούσατε να παρέχετε την αγγλική μετάφραση των προτάσεων που ακολουθούν.",
"Παρακαλούμε να υποβάλετε την αγγλική διερμηνεία για τις επόμενες προτάσεις.",
"Παρακαλώ διαθέστε την αγγλική μετάφραση για τις προτάσεις που αναφέρονται.",
"Μπορείτε να αποκαλύψετε την αγγλική μετάφραση των προσεχών προτάσεων;",


    ]

}

def add_one_example(f, src, tgt):
    for instruction_lang, source_lang, target_lang, s, t in [
        (srcshortlang, srclonglang, longlang, src, tgt),
        # (shortlang, longlang, srclonglang, tgt, src),
    ]:
        instruction = random.choice(_INSTRUCTIONS[instruction_lang]).format_map({'source_lang': source_lang, 'target_lang': target_lang})
        f.write('\n' + json.dumps(dict(instruction=instruction, input=s, output=t)))

def add_one_example_para(f, src, tgt):
    for instruction_lang, source_lang, target_lang, s, t in [
        (srcshortlang, srclonglang, longlang, src, tgt),
    ]:
        en_instruction = random.choice(_INSTRUCTIONS[instruction_lang]).format_map({'source_lang': source_lang, 'target_lang': target_lang})
    # for instruction_lang, source_lang, target_lang, s, t in [
    #     (shortlang, longlang, srclonglang, tgt, src),
    # ]:
    #     instruction = random.choice(_INSTRUCTIONS[instruction_lang]).format_map({'source_lang': source_lang, 'target_lang': target_lang})
    # We focus on to separate en_output and orig output
    f.write(json.dumps(dict(instruction=en_instruction, input=src, output=tgt, en_instruction="", en_input=tgt, en_output=src)) + '\n')

def add_one_example_plug(f, src, tgt):
    for instruction_lang, source_lang, target_lang, s, t in [
        (srcshortlang, srclonglang, longlang, src, tgt),
    ]:
        en_instruction = random.choice(_INSTRUCTIONS[instruction_lang]).format_map({'source_lang': source_lang, 'target_lang': target_lang})
    # for instruction_lang, source_lang, target_lang, s, t in [
    #     (shortlang, longlang, srclonglang, tgt, src),
    # ]:
    #     instruction = random.choice(_INSTRUCTIONS[instruction_lang]).format_map({'source_lang': source_lang, 'target_lang': target_lang})
    # We focus on to separate en_output and orig output
    f.write(json.dumps(dict(instruction=en_instruction, input=src, output=tgt, en_instruction="", en_input=tgt, en_output=src,
                            lang=longlang, trans=1)) + '\n')

def open_parallel_examples(src_f, tgt_f):
    src_lines = open(src_f).readlines()
    tgt_lines = open(tgt_f).readlines()
    for src, tgt in zip(src_lines, tgt_lines):
        yield src, tgt

wmt_dict = {
    "hi": "wmt14",
    "tr": "wmt18"
}
from datasets import load_dataset

srcshortlang, srcflorescode, srclang3code, srclonglang = 'en', 'eng', 'eng', "English"
for shortlang, florescode, lang3code, longlang in [
    # ("ar", "ara", "arb", "Arabic"),
    # # ("hi", "hin", "hin", "Hindi"),
    # ("sw", "swh", "swa", "Swahili"),
    # ("th", "tha", "tha", "Thai"),
    # # ("tr", "tur", "tur", "Turkish"),
    # ("bn", "ben", "ben", "Bengali"),
    # ("ta", "tam", "tam", "Tamil"),
    # ("te", "tel", "tel", "Telugu"),
    # ("el", "ell", "ell", "Greek"),

    ("gu", "guj", "guj", "Gujarati"),
    ("kn", "kan", "kan", "Kannada"),
    ("ml", "mal", "mal", "Malayalam"),
    ("mr", "mar", "mar", "Marathi"),
    ("pa", "pan", "pan", "Panjabi"),
]:
    # shutil.copy(f"alpaca_{shortlang}", f"trans_part_{shortlang}.jsonl")
    # with open(f"trans_part_{shortlang}.jsonl", 'a') as f:
    #     for src, tgt in open_parallel_examples(f"flores101_dataset/dev/{srcflorescode}.dev",
    #                                            f"flores101_dataset/dev/{florescode}.dev"):
    #         add_one_example(f, src, tgt)
    #
    #     for src, tgt in open_parallel_examples(f"flores101_dataset/devtest/{srcflorescode}.devtest",
    #                                            f"flores101_dataset/devtest/{florescode}.devtest"):
    #         add_one_example(f, src, tgt)
    #     for src, tgt in open_parallel_examples(f"NTREX/NTREX-128/newstest2019-src.{srclang3code}.txt",
    #                                            f"NTREX/NTREX-128/newstest2019-ref.{lang3code}.txt"):
    #         add_one_example(f, src, tgt)
    #     # if shortlang in wmt_dict:
    #     #     wmt_d = load_dataset(wmt_dict[shortlang], f"{shortlang}-{srcshortlang}", trust_remote_code=True)
    #     #     for d in wmt_d['validation']['translation']:
    #     #         add_one_example(f, d[srcshortlang], d[shortlang])
    #     #     for d in wmt_d['test']['translation']:
    #     #         add_one_example(f, d[srcshortlang], d[shortlang])

    with open(f"para_{shortlang}.jsonl", 'w') as f:
        for en_data, tgt_data in zip(open("alpaca_en"), open(f"alpaca_{shortlang}")):
            en_data, tgt_data = json.loads(en_data), json.loads(tgt_data)
            for k, v in en_data.items():
                tgt_data[f"en_{k}"] = v
            f.write(json.dumps(tgt_data) + '\n')


        for src, tgt in open_parallel_examples(f"flores101_dataset/dev/{srcflorescode}.dev",
                                               f"flores101_dataset/dev/{florescode}.dev"):
            add_one_example_para(f, src, tgt)

        for src, tgt in open_parallel_examples(f"flores101_dataset/devtest/{srcflorescode}.devtest",
                                               f"flores101_dataset/devtest/{florescode}.devtest"):
            add_one_example_para(f, src, tgt)
        for src, tgt in open_parallel_examples(f"NTREX/NTREX-128/newstest2019-src.{srclang3code}.txt",
                                               f"NTREX/NTREX-128/newstest2019-ref.{lang3code}.txt"):
            add_one_example_para(f, src, tgt)
        if shortlang in wmt_dict:
            wmt_d = load_dataset(wmt_dict[shortlang], f"{shortlang}-{srcshortlang}", trust_remote_code=True)
            for d in wmt_d['validation']['translation']:
                add_one_example_para(f, d[srcshortlang], d[shortlang])
            for d in wmt_d['test']['translation']:
                add_one_example_para(f, d[srcshortlang], d[shortlang])

    # with open(f"plug_{shortlang}.jsonl", 'w') as f:
    #     for en_data, tgt_data in zip(open("alpaca_en"), open(f"alpaca_{shortlang}")):
    #         en_data, tgt_data = json.loads(en_data), json.loads(tgt_data)
    #         for k, v in en_data.items():
    #             tgt_data[f"en_{k}"] = v
    #             tgt_data['lang'] = longlang
    #             tgt_data['trans'] = 0
    #         f.write(json.dumps(tgt_data) + '\n')
    #
    #     for src, tgt in open_parallel_examples(f"flores101_dataset/dev/{srcflorescode}.dev",
    #                                            f"flores101_dataset/dev/{florescode}.dev"):
    #         add_one_example_plug(f, src, tgt)
    #
    #     for src, tgt in open_parallel_examples(f"flores101_dataset/devtest/{srcflorescode}.devtest",
    #                                            f"flores101_dataset/devtest/{florescode}.devtest"):
    #         add_one_example_plug(f, src, tgt)
    #     for src, tgt in open_parallel_examples(f"NTREX/NTREX-128/newstest2019-src.{srclang3code}.txt",
    #                                            f"NTREX/NTREX-128/newstest2019-ref.{lang3code}.txt"):
    #         add_one_example_plug(f, src, tgt)
    #     if shortlang in wmt_dict:
    #         wmt_d = load_dataset(wmt_dict[shortlang], f"{shortlang}-{srcshortlang}", trust_remote_code=True)
    #         for d in wmt_d['validation']['translation']:
    #             add_one_example_plug(f, d[srcshortlang], d[shortlang])
    #         for d in wmt_d['test']['translation']:
    #             add_one_example_plug(f, d[srcshortlang], d[shortlang])
