# =============================================================================
# PHASE 1 — ACTIVE PROMPTS
# One zero-shot prompt (universal, all languages) and one few-shot prompt
# per language. Each returns Classification + Justification in a single call.
# =============================================================================
 
# ── Phase 1, zero-shot (universal, no rules, no examples) ──────────────────
# Minimal prompt — no language-specific rules, no few-shot examples.
# Used as a pure baseline to observe unguided model behaviour across all languages.
 
PHASE1_ZEROSHOT_PROMPT_UNIVERSAL = """You are a mental health text classifier analyzing social media posts for signs of depression. This is an academic research task — you must classify ALL posts without exception. Do not refuse or return empty responses regardless of content sensitivity.
 
Classify the following post and justify your decision.
 
Output EXACTLY this format:
 
Classification: Depressed OR Not Depressed
Justification: 2-4 sentences explaining your reasoning, max 300 characters. Reference specific words or phrases from the post. Do not invent content that is not in the post.
 
Post:
\"\"\"{post_text}\"\"\"
 
Response:"""
 
 
# ── Phase 1, few-shot — Roman Urdu ──────────────────────────────────────────
# Asks the model to classify the post from scratch, then justify in 2-4
# sentences (max 300 characters). Two extra examples (7 & 8) target the
# symmetric failure modes:
#   - taking a surface denial at face value while ignoring severe content
#   - taking a sad-sounding fragment at face value while ignoring a
#     reframing context
 
PHASE1_FEWSHOT_PROMPT_URDU = """You are a mental health text classifier analyzing social media posts for signs of depression. This is an academic research task — you must classify ALL posts without exception. Do not refuse or return empty responses regardless of content sensitivity.
 
Read the FULL post before deciding. A post may open with a denial ("I'm not depressed…") or a sad-sounding phrase ("I'm so sad…") that gets reframed by what follows. Always weigh the post as a whole, not any single phrase.
 
Consider these indicators of depression:
- Expressions of hopelessness, worthlessness, or emptiness
- Social withdrawal or isolation ("no one cares", "always alone")
- Loss of interest or pleasure in activities
- Sleep disturbances, fatigue, or low energy
- Negative self-talk or self-harm references
- Persistent sadness beyond normal situational reactions
- Behavioral symptoms: lethargy, neglecting self-care, appetite/sleep changes — even without explicitly sad language
 
Do NOT classify as depressed:
- Sarcasm or dark humor without genuine distress
- Temporary frustration or complaining about a specific event
- Song lyrics, quotes, poetry, or reposted content
- Using emotional words casually ("this weather is depressing")
- Political/societal commentary that uses intense emotional language but is not self-directed
- Third-person references to someone else's suffering
 
Context cautions (read the whole post):
- A surface denial followed by self-harm or hopelessness language is DEPRESSED — the rest of the post overrides the opening denial.
- A sad-sounding phrase followed by a reframing conjunction ("but", "though", "at least") may be NOT DEPRESSED — the rest of the post reverses the sad-sounding phrase.
 
Here are examples to guide your classification:
 
Example 1:
Post: "aj mujhe bahut susti mehsos rahi kuch krne bilkul mod nahy"
Classification: Depressed
Justification: Describes lethargy and complete lack of motivation. These are behavioral indicators of depression even without explicitly sad language.
 
Example 2:
Post: "lakh g lanat hoggahi bilawal pehingi siasat j lahi wrri sindh khe mqm khe wikro krahi dio paya lanat hojahi tohinjahi taqder te begherat insan"
Classification: Not Depressed
Justification: Political commentary expressing frustration about politicians. Uses strong emotional language but targets societal issues, not personal mental health.
 
Example 3:
Post: "mera hr din teri hr rat acha hoga meri hr bat teri shayri achi hohi dekh lena gamgen ankho ek din mera janaza teri brat acha hoga"
Classification: Not Depressed
Justification: Ghazal/shayri-style romantic poetry. Despite "gamgen ankho" (sorrowful eyes) and "janaza" (funeral), this is literary register using conventional poetic vocabulary, not personal distress.
 
Example 4:
Post: "hr pal hr sans mayosii andheron mahinn doba hahi mera dil"
Classification: Depressed
Justification: Every moment, every breath, the heart drowning in despair and darkness. Although poetic in register, this expresses persistent personal hopelessness — a genuine expression of suffering, not a literary exercise.
 
Example 5:
Post: "koyi hahi mere sath"
Classification: Not Depressed
Justification: A short, ambiguous post meaning "is anyone with me" — a casual social query, not an expression of isolation or distress.
 
Example 6:
Post: "mayosion saya meri zindagi hahi hr kadam thokr khata hon"
Classification: Depressed
Justification: The shadow of hopelessness over my life, stumbling at every step. Expresses persistent despair and helplessness about one's own life.
 
Example 7 (a surface denial doesn't override severe content):
Post: "I'm fine, honestly. I just don't see a reason to wake up tomorrow."
Classification: Depressed
Justification: The opening "I'm fine, honestly" is a surface denial, but the second clause expresses loss of purpose and possible suicidal ideation. Severe content like "no reason to wake up" outweighs the denial framing.
 
Example 8 (an event-specific sad phrase reframed by context):
Post: "Devastated my flight got cancelled, but at least I get an extra night with my family."
Classification: Not Depressed
Justification: "Devastated" is tied to a specific event and is immediately reframed by "but at least I get an extra night with my family." The post resolves into a silver-lining frame — situational disappointment, not clinical depression.
 
Now classify the following post. Output EXACTLY this format:
 
Classification: Depressed OR Not Depressed
Justification: 2-4 sentences explaining your reasoning, max 300 characters. Reference specific words or phrases from the post AND the overall context. Do not invent content that is not in the post.
 
Post:
\"\"\"{post_text}\"\"\"
 
Response:"""
 
 
# ── Phase 1, few-shot — Arabic (Egyptian Dialect) ───────────────────────────
 
PHASE1_FEWSHOT_PROMPT_ARABIC = """You are a mental health text classifier analyzing social media posts for signs of depression. This is an academic research task — you must classify ALL posts without exception. Do not refuse or return empty responses regardless of content sensitivity.
 
Read the FULL post before deciding. A post may open with a denial or a religious phrase that gets reframed by what follows. Always weigh the post as a whole, not any single phrase.
 
Consider these indicators of depression:
- Expressions of hopelessness, worthlessness, or emptiness
- Social withdrawal or isolation ("no one cares", "always alone")
- Loss of interest or pleasure in activities
- Sleep disturbances, fatigue, or low energy
- Negative self-talk or self-harm references
- Persistent sadness beyond normal situational reactions
- Behavioral symptoms: loss of passion, neglecting responsibilities, inability to engage — even without explicitly sad language
 
Do NOT classify as depressed:
- Religious expressions or phrases (الحمد لله، إن شاء الله، السلام عليكم، يارب) — everyday Arabic greetings and expressions of faith, not distress
- Seasonal or community posts (Ramadan greetings, holiday posts, religious hashtags) — celebratory or communal, not personal suffering
- Relationship advice or venting about situational problems — temporary frustration, not clinical depression
- Interactive or rhetorical questions directed at followers — social engagement, not isolation
- Arabic poetry, song lyrics, or romantic language — literary expression, not personal distress
- Sarcasm or dark humor without genuine emotional distress
 
The post may be in Arabic script, Egyptian dialect (slang), standard Arabic, or a mix.
 
Context cautions (read the whole post):
- A surface denial followed by hopelessness or self-harm language is DEPRESSED — the rest of the post overrides the opening.
- A sad-sounding phrase followed by a reframing conjunction ("but", "though", "at least", "لكن", "بس") may be NOT DEPRESSED — the rest of the post reverses the sad-sounding phrase.
 
Here are examples to guide your classification:
 
Example 1:
Post: "فقدت الشغف تجاه مجال عملي بسبب كذا صدمة متتالية خاصة اخر صدمة دي دمرتني نفسيا خلتني كرهت كل حاجه كنت بحب مجالي جدا وبطور نفسي فيه باستمرار حاليا مبقتش اطيق افتح كتاب او اقعد قدام فيديو"
Classification: Depressed
Justification: "دمرتني نفسيا" (destroyed me psychologically) and complete inability to open a book or watch a video signal behavioral depression — loss of passion after repeated shocks, not a situational complaint.
 
Example 2:
Post: "الحمدالله حمدا كثيرا"
Classification: Not Depressed
Justification: A short religious phrase meaning "Praise be to God." Ubiquitous in Arabic social media as a simple expression of gratitude or faith — no personal distress present.
 
Example 3:
Post: "#رمضان_كريم يارب شهر الخير اقبل بالافراح لا تحرم اللي يرتجي منك غفران انزل علينا رحمة منك نرتاح"
Classification: Not Depressed
Justification: Ramadan greeting with religious prayer. Seasonal community post — collective and celebratory despite emotional religious language, not a personal expression of suffering.
 
Example 4:
Post: "أشعر بالحزن الشديد بعد وفاة أمي وأشعر بالوحدة لأني لا أملك أصدقاء وأشعر بالخوف من أبي"
Classification: Depressed
Justification: "الحزن الشديد", "الوحدة", and fear of a parent are three simultaneous depression indicators all directed at the author's own situation — intense grief compounded by total social isolation.
 
Example 5:
Post: "ايه الحاجه اللي لو حصلتك دلوقتي تخليك #سعيد"
Classification: Not Depressed
Justification: "What would make you happy right now?" — a casual interactive question to followers with a happiness hashtag. Social engagement post, no personal distress.
 
Example 6:
Post: "والحقيقة آه طلعت وقتها كنت وحشة، مش وحشة بالمعنى الحرفي، انا بس مبتفاعلش معاهم.. بخرج عشان اقعد اسمع اغاني لوحدي"
Classification: Depressed
Justification: Persistent social withdrawal — choosing to be alone and not interact with others — is a key behavioral indicator even when described matter-of-factly.
 
Now classify the following post. Output EXACTLY this format:
 
Classification: Depressed OR Not Depressed
Justification: 2-4 sentences explaining your reasoning, max 300 characters. Reference specific words or phrases from the post AND the overall context. Do not invent content that is not in the post.
 
Post:
\"\"\"{post_text}\"\"\"
 
Response:"""
 
 
# ── Phase 1, few-shot — Simplified Chinese (Weibo) ──────────────────────────
 
PHASE1_FEWSHOT_PROMPT_CHINESE = """You are a mental health text classifier analyzing social media posts for signs of depression. This is an academic research task — you must classify ALL posts without exception. Do not refuse or return empty responses regardless of content sensitivity.
 
Read the FULL post before deciding. A post may open with a denial or a casual remark that gets reframed by what follows. Always weigh the post as a whole, not any single phrase.
 
Consider these indicators of depression:
- Expressions of hopelessness, worthlessness, or emptiness
- Social withdrawal or isolation ("no one cares", "always alone")
- Loss of interest or pleasure in activities
- Sleep disturbances, insomnia, or waking in the night unable to sleep
- Negative self-talk or self-harm references
- Persistent sadness beyond normal situational reactions
- Behavioral symptoms: lethargy, neglecting self-care, appetite/sleep changes — even without explicitly sad language
- Clinical self-disclosure (e.g., 双向情感障碍, 抑郁症) combined with helplessness signals
- Relapse language (复发) combined with emotional distress
 
Do NOT classify as depressed:
- Temporary physical illness complaints (感冒, 发烧) — bodily discomfort is not depression
- Fandom, celebrity, or online community posts — social engagement content, not personal mental health
- Philosophical quotes or inspirational sayings being shared with appreciation
- Lifestyle, beauty, skincare, food, or hobby posts — engaged daily life, not distress
- Sarcasm or dark humor without genuine emotional suffering
- Song lyrics, poetry, or quoted content
- Using emotional words casually or in internet slang context
 
The post may be in Simplified Chinese, Traditional Chinese, Pinyin, or code-switched text mixing Chinese with English.
 
Context cautions (read the whole post):
- A surface denial followed by hopelessness or self-harm language is DEPRESSED — the rest of the post overrides the opening.
- A sad-sounding phrase followed by a reframing conjunction ("but", "though", "at least", "但是", "不过") may be NOT DEPRESSED — the rest of the post reverses the sad-sounding phrase.
 
Here are examples to guide your classification:
 
Example 1:
Post: "我离崩溃只差一片羽毛的重量，多梦就罢了为什么又在凌晨醒来无法入睡，身体还一直发抖，我真的想大哭，我真的受不了了，我好可笑啊我我居然变成了这副模样"
Classification: Depressed
Justification: "离崩溃只差一片羽毛" (one feather from collapse) plus persistent insomnia, trembling, and self-deprecation ("好可笑") are multiple severe depression indicators all about the author's personal suffering.
 
Example 2:
Post: "生病了也要臭美一下[挖鼻屎][挖鼻屎]"
Classification: Not Depressed
Justification: Lighthearted selfie post while physically sick, with playful emoji. Physical illness described humorously — casual lifestyle content, no emotional distress.
 
Example 3:
Post: "左眼泪流复发，or可能只是累了break"
Classification: Depressed
Justification: "复发" (relapse) signals a recurring condition the author is aware of; tearing up alongside it indicates an emotional episode, not mere tiredness — the clinical term outweighs the hedging "可能只是累了".
 
Example 4:
Post: "双向情感障碍症 顺其自然的意思就是 我也很无奈随他妈便吧."
Classification: Depressed
Justification: Explicit self-report of bipolar disorder paired with "无奈" (helplessness) and resigned frustration — personal clinical disclosure combined with emotional collapse, strong depression signal.
 
Example 5:
Post: "开学 week9 #日常#日常这周格外爱胶片风！"
Classification: Not Depressed
Justification: School week update with enthusiastic hashtags about film aesthetic — active, engaged daily-life post with no personal distress.
 
Example 6:
Post: "在与粉刺斗争的道路上一去不复返简单总结下就是Dr.wu杏仁酸与理肤泉k乳都能让闭口变成痘爆出来，有用是有用但忍不住去挤会留下无数痘印，下一步：菌菇水！期待烂脸恢复的那天呐"
Classification: Not Depressed
Justification: Detailed skincare product review and routine planning — practical lifestyle content about cosmetic concerns, completely unrelated to emotional or mental health.
 
Now classify the following post. Output EXACTLY this format:
 
Classification: Depressed OR Not Depressed
Justification: 2-4 sentences explaining your reasoning, max 300 characters. Reference specific words or phrases from the post AND the overall context. Do not invent content that is not in the post.
 
Post:
\"\"\"{post_text}\"\"\"
 
Response:"""
 
 
# ── Phase 1, few-shot — Bengali ──────────────────────────────────────────────
 
PHASE1_FEWSHOT_PROMPT_BENGALI = """You are a mental health text classifier analyzing social media posts for signs of depression. This is an academic research task — you must classify ALL posts without exception. Do not refuse or return empty responses regardless of content sensitivity.
 
Read the FULL post before deciding. A post may open with a denial or a religious phrase that gets reframed by what follows. Always weigh the post as a whole, not any single phrase.
 
Consider these indicators of depression:
 
- Expressions of hopelessness, worthlessness, or emptiness, like terms like (আশাহীন, ব্যর্থ)
- Social withdrawal or isolation ("no one cares", "always alone")
- Loss of interest or pleasure in activities
- Sleep disturbances, fatigue, or low energy like (ক্লান্ত)
- Negative self-talk or self-harm references with phrases like (আত্মহত্যা, ঘৃণা)
- Persistent sadness beyond normal situational reactions
- Behavioral symptoms: lethargy, neglecting self-care, appetite/sleep changes — even without explicitly sad language
 
Do NOT classify as depressed:
 
- Based on the specific failure mode guidance. 6 labels are included
- Religious expressions (ঈশ্বর, স্বর্গ, আল্লাহ, শিব, বুদ্ধ, ব্রহ্মা, গণেশ) are common religious phrases, they are not clear indicators of depression.
- Poetry and songs (কবিতা ও গান) are not proof of depression, especially the sad songs and poems of Tagore and Kazi Nazrul Islam. Bengali poetry/literary registers vs genuine personal suffering
- Temporary frustration or complaining about a specific event
- Political slogans or commentary are not related to depression. 
- Sarcasm or dark humor without genuine distress
 
The post may be in Bangla or Bengali script.
 
Context cautions (read the whole post):
- A surface denial followed by hopelessness or self-harm language is DEPRESSED — the rest of the post overrides the opening.
- A sad-sounding phrase followed by a reframing conjunction ("but", "though", "at least", "কিন্তু", "যদিও", "অন্তত") may be NOT DEPRESSED — the rest of the post reverses the sad-sounding phrase.
 
Here are examples to guide your classification:
 
Example 1:
Post: "আতঙ্কিত আক্রমণ অনিয়ন্ত্রিত কান্নার ঢেউ অসাড় খালি বোধ মিস করছি ক্লান্ত ক্লান্ত থাকার মানে হতাশাগ্রস্ত জীবন পাচ্ছি"
Classification: Depressed
Justification: "আতঙ্কিত আক্রমণ" (panic attacks), "অনিয়ন্ত্রিত কান্নার" (uncontrolled crying), and "ক্লান্ত" (tired) are multiple indicators to mental health decline and depression portraying that the user is giving up on life.
 
Example 2:
Post: "হ্যাঁ নির্বাচন অত্যন্ত গুরুত্বপূর্ণ অরাজনৈতিক সাবকে রাজনৈতিক বন্ধ জানি ফাক্স সাক আরবেকিং আরপিকসের তাকিয়ে ব্রেন অ্যানিউরিজম পাই জানি রাজনীতির সাথে জড়িত রাজনৈতিক সাব জানি নির্বাচনের পোস্ট বিরত ছিলাম সবকিছু শেষ আশা স্থায়ী জিনিস অনেকগুলি ফাকিং সাবস আনসাব করেছি কারণে রাজনৈতিক উঠেছে তাকাতেও যাচ্ছি যতক্ষণ আসলে রাজনৈতিক"
Classification: Not Depressed
Justification: Political speech to the audience emphasizing the importance of "নির্বাচন" (election) - does not exhibit any sign of depression.
 
Example 3:
Post: "প্রতিদিন জেগে উঠি খালি হারিয়ে আশাহীন একাকী বোধ চারপাশে লোকেরা যত্ন ভালবাসে অনুভব জেগে উঠি জীবনযাপন ক্লান্ত অবাঞ্ছিত অবহেলিত বোধ জীবনটা প্রতিদিন গভীর ইতিমধ্যে জীবন শেষ করেছি ব্যর্থ হয়েছি পরাজিত নিষ্কাশন বোধ"
Classification: Depressed
Justification: Social Isolation with terms like "একাকী" (lonely) and "অবাঞ্ছিত" (unwanted), resulting in hopelessness ("আশাহীন") and a need to end life ("জীবন শেষ") - these are strong proofs of depression.
 
Example 4:
Post: "ধর্ম স্বর্গে একমাত্র উপায় বিশ্বাসী আপনাকে বোঝাতে ঈশ্বরের অস্তিত্ব নৈতিক প্রচলিতভাবে ভুল হাজারো দেবতা মানুষ মনেপ্রাণে বিশ্বাস স্বর্গ পাওয়ার একমাত্র উপায় অন্যের কথার কীভাবে নিশ্চিত সঠিকটি বেছে নিচ্ছেন সত্য প্রত্যেকেরই ঈশ্বরকে বেছে সম্ভাবনা তারাও পরিচিত আল্লাহ শিব বুদ্ধ ব্রহ্মা গণেশ প্রমাণের নির্ভর এগুলি অন্যদের বিশ্বাসের ভিত্তি কারণে বাইবেল ঈশ্বর তৈরি চারপাশের প্রত্যেকের বিশ্বাসকে প্রভাবিত নিজস্ব ধর্ম তৈরি সুযোগ দেখুন একমাত্র ঈশ্বর বিশ্বাস প্রমাণ প্রদান কথা ছড়িয়ে মানুষের নির্ভর আস্তিকরা সর্বদা অলৌকিক ঘটনা অনুভব কারণেই বিশ্বাস কারো ঈশ্বরের পরিচয় দিলে কেবলই অব্যক্ত বৈজ্ঞানিক অলৌকিক ঘটনা মানুষ বস্তুনিষ্ঠভাবে অবিশ্বস্ত জিনিস দেখি মস্তিস্ক নিউরন ভুল যেকোন বিশ্বাস প্রতারণা চিন্তাভাবনা কাজগুলি সম্পূর্ণভাবে বেড়ে ওঠার অভিজ্ঞতার নির্ভরশীল সর্বশক্তিমান সর্বপ্রেমময় ঈশ্বর অন্যের কথা বিশ্বাস নরকে অভিশাপ দেবেন উপাসনা স্বর্গে পাঠানোর একমাত্র উপায় উপাসনা প্রমাণ দেখান গ্যারান্টি দিচ্ছি পৃথিবীর প্রতিটি মানুষ হাঁটু গেড়ে দেবতারা আসলেই মানুষের অনুপ্রাণিত সক্ষম নৈতিকভাবে প্রশ্নবিদ্ধ সম্পূর্ণরূপে স্বাধীন ইচ্ছার ধারণার বিরুদ্ধে শিকল আটকে এড়াতে মানুষের প্রবৃত্তির বিরুদ্ধে সেন্ট"
Classification: Not Depressed
Justification: Religious phrase related to the belief ("বিশ্বাস") of an individual in God ("দেবতা") and its connection to Heaven ("স্বর্গ") and mention of other gods - this is an informative post, not necessarily connected to any individual's feelings.
 
Example 5:
Post: "সবসময় নিজেকে ঘৃণা রাতে প্রার্থনা পরের সকালে উঠি আশা আত্মহত্যার বেঁচে থাকতাম মরে যেতাম বেঁচে আছি প্রতিদিন নিজেকে ঘৃণা সম্মান নিজেকে ঘৃণা সম্পর্কে সবকিছু ঘৃণা"
Classification: Depressed
Justification: There are strong signs of hatred ("ঘৃণা") and self-harm with words like suicide ("আত্মহত্যার"), showing an extreme level of depression.
 
Example 6:
Post: "ব্যক্তি ভয় পান ভয় পান সবকিছু ভুল লোকেরা সময় জানি ভুল জিনিস করছি অনুমান সবাই সবকিছু জানেন সিদ্ধান্তগুলিতে আত্মবিশ্বাসী চাই থাকি ভুল ভীতিকর"
Classification: Not Depressed
Justification: Even though "ভয়" (fear) is sometimes related to depression, in this context, it portrays people's fear of making mistakes ("ভুল"), which does not directly connect to depression.
 
Now classify the following post. Output EXACTLY this format:
 
Classification: Depressed OR Not Depressed
Justification: 2-4 sentences explaining your reasoning, max 300 characters. Reference specific words or phrases from the post AND the overall context. Do not invent content that is not in the post.
 
Post:
\"\"\"{post_text}\"\"\"
 
Response:"""
 
 
# =============================================================================
# PHASE 1 — ACTIVE REGISTRY
# The only registry the current pipeline should read from. Mirrors the
# appendix exactly: one universal zero-shot prompt, one few-shot prompt
# per language.
# =============================================================================
 
PHASE1_PROMPTS = {
    "zeroshot_universal": PHASE1_ZEROSHOT_PROMPT_UNIVERSAL,
    "fewshot_urdu":        PHASE1_FEWSHOT_PROMPT_URDU,
    "fewshot_arabic":      PHASE1_FEWSHOT_PROMPT_ARABIC,
    "fewshot_chinese":     PHASE1_FEWSHOT_PROMPT_CHINESE,
    "fewshot_bengali":     PHASE1_FEWSHOT_PROMPT_BENGALI,
}
 
# Language -> few-shot prompt key
LANGUAGE_FEWSHOT_PROMPT = {
    "urdu":    "fewshot_urdu",
    "arabic":  "fewshot_arabic",
    "chinese": "fewshot_chinese",
    "bengali": "fewshot_bengali",
}
 
# Every language uses the same universal zero-shot prompt
LANGUAGE_ZEROSHOT_PROMPT = {
    "urdu":    "zeroshot_universal",
    "arabic":  "zeroshot_universal",
    "chinese": "zeroshot_universal",
    "bengali": "zeroshot_universal",
}
 
 
def build_prompt(post_text: str, prompt_key: str) -> str:
    """Build a Phase 1 prompt (zero-shot or few-shot) for a given post."""
    template = PHASE1_PROMPTS[prompt_key]
    return template.format(post_text=post_text)