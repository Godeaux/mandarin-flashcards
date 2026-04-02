/* ============================================
   漢字 Study — Mandarin SRS Webapp
   Pure JS, no dependencies
   ============================================ */

// ==========================================
// CARD DATA — 31 curated beginner characters
// ==========================================

const DEFAULT_DECK = [
  // --- Nature ---
  {
    id: "shan1", char: "山", pinyin: "shān", meaning: "mountain",
    category: "nature",
    mnemonic: "Three peaks rising up — it literally looks like a mountain range! The middle peak is tallest, with two smaller ones on each side.",
    visual: "⛰️  /|\\ \n /|\\ ",
    soundBridge: "\"shān\" sounds like \"shun\" — you shun the difficult climb up the mountain",
    radicals: "Radical: 山 (mountain) — this IS the radical",
    exampleSentence: "我喜欢爬山。",
    exampleBreakdown: "I (我 wǒ) like (喜欢 xǐhuān) climbing (爬 pá) mountains (山 shān)"
  },
  {
    id: "shui3", char: "水", pinyin: "shuǐ", meaning: "water",
    category: "nature",
    mnemonic: "Imagine a river flowing: the central stroke is the current, and the side strokes are splashing water droplets.",
    visual: "💧 ~≈~ 💧",
    soundBridge: "\"shuǐ\" sounds like \"shway\" — water sways in the river",
    radicals: "Radical: 水/氵(water)",
    exampleSentence: "请给我一杯水。",
    exampleBreakdown: "please (请 qǐng) give me (给我 gěiwǒ) one (一 yī) cup (杯 bēi) water (水 shuǐ)"
  },
  {
    id: "huo3", char: "火", pinyin: "huǒ", meaning: "fire",
    category: "nature",
    mnemonic: "A person with arms up in alarm, surrounded by flames! The two outer dots are sparks flying off the central blaze.",
    visual: "🔥 \\ | / 🔥",
    soundBridge: "\"huǒ\" sounds like \"hwo\" — whoah, fire!",
    radicals: "Radical: 火/灬 (fire)",
    exampleSentence: "我们去吃火锅。",
    exampleBreakdown: "we (我们 wǒmen) go (去 qù) eat (吃 chī) fire (火 huǒ) pot (锅 guō)"
  },
  {
    id: "mu4", char: "木", pinyin: "mù", meaning: "tree / wood",
    category: "nature",
    mnemonic: "A tree with its trunk going straight up, branches spreading at the top, and roots reaching down into the soil.",
    visual: "🌳  十\n / \\",
    soundBridge: "\"mù\" — \"moo,\" the cow stands under the tree",
    radicals: "Radical: 木 (tree) — this IS the radical",
    exampleSentence: "山上有很多树木。",
    exampleBreakdown: "mountain (山上 shānshàng) have (有 yǒu) very many (很多 hěnduō) trees (树木 shùmù)"
  },
  {
    id: "ri4", char: "日", pinyin: "rì", meaning: "sun / day",
    category: "nature",
    mnemonic: "A window showing the sun! The ancient form was a circle with a dot (☉), which squared off into this box with a line through it.",
    visual: "☀️ [—] ☀️",
    soundBridge: "\"rì\" sounds like \"ree\" — the sun makes you say \"really bright!\"",
    radicals: "Radical: 日 (sun) — this IS the radical",
    exampleSentence: "今日是个好日子。",
    exampleBreakdown: "today (今日 jīnrì) is (是 shì) a good (好 hǎo) day (日子 rìzi)"
  },
  {
    id: "yue4", char: "月", pinyin: "yuè", meaning: "moon / month",
    category: "nature",
    mnemonic: "A crescent moon! The opening on the side lets moonlight spill through. Also means 'month' because the moon cycles monthly.",
    visual: "🌙 ( ) 🌙",
    soundBridge: "\"yuè\" sounds like \"you-eh\" — \"you, eh, look at that moon!\"",
    radicals: "Radical: 月 (moon/flesh)",
    exampleSentence: "雨停了，我看到月亮。",
    exampleBreakdown: "rain (雨 yǔ) stopped (停了 tíngle) I (我 wǒ) see (看到 kàndào) moon (月亮 yuèliàng)"
  },
  {
    id: "yu3", char: "雨", pinyin: "yǔ", meaning: "rain",
    category: "nature",
    mnemonic: "The top horizontal line is the sky, the box below is a cloud, and the four dots are raindrops falling down!",
    visual: "☁️ . . . . 🌧️",
    soundBridge: "\"yǔ\" sounds like \"you\" — the rain falls on YOU",
    radicals: "Radical: 雨 (rain) — this IS the radical",
    exampleSentence: "突然下雨了。",
    exampleBreakdown: "suddenly (突然 tūrán) start (下 xià) rain (雨 yǔ)"
  },
  {
    id: "tian2", char: "田", pinyin: "tián", meaning: "field",
    category: "nature",
    mnemonic: "A bird's-eye view of rice paddies divided into four neat plots by paths crossing through them.",
    visual: "🌾 [+] 🌾",
    soundBridge: "\"tián\" sounds like \"tee-en\" — tee off into the open field",
    radicals: "Radical: 田 (field) — this IS the radical",
    exampleSentence: "山下是绿色的田。",
    exampleBreakdown: "below the mountain (山下 shānxià) is (是 shì) green (绿色的 lǜsède) field (田 tián)"
  },
  {
    id: "shi2", char: "石", pinyin: "shí", meaning: "stone / rock",
    category: "nature",
    mnemonic: "A cliff face (厂) with a solid mouth-shaped rock (口) sitting at its base. Stone cold and immovable!",
    visual: "🪨  厂\n    口",
    soundBridge: "\"shí\" sounds like \"sure\" — solid as a rock, for sure!",
    radicals: "Radical: 石 (stone) — this IS the radical",
    exampleSentence: "路上有大石头。",
    exampleBreakdown: "road (路上 lùshàng) have (有 yǒu) big (大 dà) stone (石头 shítou)"
  },
  {
    id: "feng1", char: "风", pinyin: "fēng", meaning: "wind",
    category: "nature",
    mnemonic: "An outer frame like a sail catching the wind, with a swirl (㐅) of air spinning inside it.",
    visual: "🌬️ 〔 × 〕💨",
    soundBridge: "\"fēng\" sounds like \"fung\" — the wind goes whoooosh through the fungus forest",
    radicals: "Simplified from 風. Contains 几 (table/bench frame)",
    exampleSentence: "山上的风很大。",
    exampleBreakdown: "mountaintop (山上的 shānshàngde) wind (风 fēng) very (很 hěn) strong (大 dà)"
  },

  // --- People/Body ---
  {
    id: "ren2", char: "人", pinyin: "rén", meaning: "person / people",
    category: "people",
    mnemonic: "A person walking! Two legs mid-stride, leaning forward with purpose. The simplest picture of a human in motion.",
    visual: "🚶 / \\",
    soundBridge: "\"rén\" sounds like \"run\" — a person running",
    radicals: "Radical: 人/亻 (person) — this IS the radical",
    exampleSentence: "街上人很多。",
    exampleBreakdown: "street (街上 jiēshàng) people (人 rén) very (很 hěn) many (多 duō)"
  },
  {
    id: "da4", char: "大", pinyin: "dà", meaning: "big / large",
    category: "people",
    mnemonic: "A person (人) stretching their arms out wide to show how BIG something is! \"It was THIS big!\"",
    visual: "🙆 — | —",
    soundBridge: "\"dà\" sounds like \"dah\" — \"Ta-dah! Look how BIG!\"",
    radicals: "Radical: 大 (big) — person 人 with arms spread",
    exampleSentence: "这个城市很大。",
    exampleBreakdown: "this (这个 zhège) city (城市 chéngshì) very (很 hěn) big (大 dà)"
  },
  {
    id: "xiao3", char: "小", pinyin: "xiǎo", meaning: "small / little",
    category: "people",
    mnemonic: "A tiny thing being split into even smaller pieces — the vertical line divides, and the two dots are the tiny fragments flying apart.",
    visual: "🤏 ·|·",
    soundBridge: "\"xiǎo\" sounds like \"sh-ow\" — \"shh, it's so small!\"",
    radicals: "Radical: 小 (small)",
    exampleSentence: "我找到一个小店。",
    exampleBreakdown: "I (我 wǒ) find (找到 zhǎodào) one (一个 yígè) small (小 xiǎo) shop (店 diàn)"
  },
  {
    id: "kou3", char: "口", pinyin: "kǒu", meaning: "mouth",
    category: "people",
    mnemonic: "An open mouth, ready to speak! A perfect square opening. Simple as that — open wide and say \"ahh\"!",
    visual: "👄 [ ] 😮",
    soundBridge: "\"kǒu\" sounds like \"co\" — open your mouth and co-mmunicate",
    radicals: "Radical: 口 (mouth) — this IS the radical",
    exampleSentence: "门口有人在等。",
    exampleBreakdown: "doorway (门口 ménkǒu) have (有 yǒu) people (人 rén) waiting (在等 zàiděng)"
  },
  {
    id: "shou3", char: "手", pinyin: "shǒu", meaning: "hand",
    category: "people",
    mnemonic: "The horizontal lines are fingers, and the vertical stroke with a hook is the palm and wrist. A hand reaching out!",
    visual: "✋ ≡ |",
    soundBridge: "\"shǒu\" sounds like \"show\" — show me your hand!",
    radicals: "Radical: 手/扌 (hand) — this IS the radical",
    exampleSentence: "我用手拿筷子。",
    exampleBreakdown: "I (我 wǒ) use (用 yòng) hand (手 shǒu) hold (拿 ná) chopsticks (筷子 kuàizi)"
  },
  {
    id: "mu4b", char: "目", pinyin: "mù", meaning: "eye",
    category: "people",
    mnemonic: "An eye turned sideways! The ancient form was a clear eye shape 👁, which got squared off. The lines inside are the iris and pupil.",
    visual: "👁️ [=] 👁️",
    soundBridge: "\"mù\" — \"moo,\" the cow looks at you with big eyes",
    radicals: "Radical: 目 (eye) — this IS the radical",
    exampleSentence: "我目不转睛地看。",
    exampleBreakdown: "I (我 wǒ) eye (目 mù) not (不 bù) turn (转 zhuǎn) stare (睛 jīng)"
  },
  {
    id: "nv3", char: "女", pinyin: "nǚ", meaning: "woman",
    category: "people",
    mnemonic: "A graceful figure kneeling with arms crossed — the ancient Chinese depiction of a woman in a dignified pose.",
    visual: "💃 く ノ",
    soundBridge: "\"nǚ\" sounds like \"new\" — a new woman in your life",
    radicals: "Radical: 女 (woman) — this IS the radical",
    exampleSentence: "那个女人是导游。",
    exampleBreakdown: "that (那个 nàgè) woman (女人 nǚrén) is (是 shì) tour guide (导游 dǎoyóu)"
  },
  {
    id: "zi3", char: "子", pinyin: "zǐ", meaning: "child / son",
    category: "people",
    mnemonic: "A baby in a swaddling blanket with arms reaching out! The top is the head, the horizontal line is outstretched arms, and the curve is the wrapped body.",
    visual: "👶 了 +arms",
    soundBridge: "\"zǐ\" sounds like \"zuh\" — a child going \"zzzz\" while sleeping",
    radicals: "Radical: 子 (child)",
    exampleSentence: "她带着孩子。",
    exampleBreakdown: "she (她 tā) brings (带着 dàizhe) child (孩子 háizi)"
  },
  {
    id: "xin1", char: "心", pinyin: "xīn", meaning: "heart / mind",
    category: "people",
    mnemonic: "The curved bottom is the heart shape, and the three dots above are the heartbeat — lub-dub-lub! In Chinese culture, heart and mind are one.",
    visual: "💗 . . . ♡",
    soundBridge: "\"xīn\" sounds like \"shin\" — your heart is in it, you feel it in your shins from running",
    radicals: "Radical: 心/忄 (heart) — this IS the radical",
    exampleSentence: "我很开心。",
    exampleBreakdown: "I (我 wǒ) very (很 hěn) happy (开心 kāixīn)"
  },

  // --- Numbers ---
  {
    id: "yi1", char: "一", pinyin: "yī", meaning: "one",
    category: "numbers",
    mnemonic: "One single horizontal stroke — the simplest character. One line = one! It couldn't be more intuitive.",
    visual: "☝️ ———",
    soundBridge: "\"yī\" sounds like \"ee\" — easy as one!",
    radicals: "Radical: 一 (one)",
    exampleSentence: "我买了一个苹果。",
    exampleBreakdown: "I (我 wǒ) bought (买了 mǎile) one (一 yī) measure word (个 gè) apple (苹果 píngguǒ)"
  },
  {
    id: "er4", char: "二", pinyin: "èr", meaning: "two",
    category: "numbers",
    mnemonic: "Two horizontal strokes stacked. Two lines = two! The top line is shorter (like a child) and the bottom is longer (like a parent).",
    visual: "✌️ ——\n———",
    soundBridge: "\"èr\" sounds like \"are\" — there ARE two of them",
    radicals: "Radical: 二 (two)",
    exampleSentence: "还要二杯茶。",
    exampleBreakdown: "also want (还要 háiyào) two (二 èr) cups (杯 bēi) tea (茶 chá)"
  },
  {
    id: "san1", char: "三", pinyin: "sān", meaning: "three",
    category: "numbers",
    mnemonic: "Three horizontal strokes — the pattern continues! Like a hamburger: top bun, patty, bottom bun. Three layers = three!",
    visual: "🥉 ——\n———\n——",
    soundBridge: "\"sān\" sounds like \"sahn\" — the sun rises three times",
    radicals: "Radical: 一 (one), repeated three times",
    exampleSentence: "一共三十块钱。",
    exampleBreakdown: "total (一共 yígòng) three (三 sān) ten (十 shí) dollars (块钱 kuàiqián)"
  },
  {
    id: "si4", char: "四", pinyin: "sì", meaning: "four",
    category: "numbers",
    mnemonic: "The box contains what looks like legs of a table — four legs! The ancient form showed four lines, but now it's stylized inside a frame.",
    visual: "4️⃣ [儿]",
    soundBridge: "\"sì\" sounds like \"suh\" — there are four S's in \"Mississippi\"",
    radicals: "Radical: 囗 (enclosure)",
    exampleSentence: "我走了四条街。",
    exampleBreakdown: "I (我 wǒ) walked (走了 zǒule) four (四 sì) measure word (条 tiáo) streets (街 jiē)"
  },
  {
    id: "wu3", char: "五", pinyin: "wǔ", meaning: "five",
    category: "numbers",
    mnemonic: "Like the Roman numeral V rotated — the crossing strokes between two horizontal lines represent five. Think of a hand giving a high-five!",
    visual: "🖐️ ═╤═",
    soundBridge: "\"wǔ\" sounds like \"woo\" — high five, woo!",
    radicals: "Radical: 二 (two) with strokes between",
    exampleSentence: "今天走了五公里。",
    exampleBreakdown: "today (今天 jīntiān) walked (走了 zǒule) five (五 wǔ) kilometers (公里 gōnglǐ)"
  },

  // --- Common ---
  {
    id: "hao3", char: "好", pinyin: "hǎo", meaning: "good / well",
    category: "common",
    mnemonic: "Woman (女) + child (子) = good! A mother with her child — the most universally good thing. Family harmony = goodness.",
    visual: "👩‍👦 女+子 = 好",
    soundBridge: "\"hǎo\" sounds like \"how\" — \"How are you?\" \"Good!\"",
    radicals: "女 (woman) + 子 (child)",
    exampleSentence: "你好！",
    exampleBreakdown: "you (你 nǐ) good (好 hǎo)"
  },
  {
    id: "zhong1", char: "中", pinyin: "zhōng", meaning: "middle / China",
    category: "common",
    mnemonic: "An arrow hitting the exact center of a target! The vertical line pierces through the middle of the box. China sees itself as the Middle Kingdom (中国).",
    visual: "🎯 [|]",
    soundBridge: "\"zhōng\" sounds like \"jung\" — right in the middle of the jungle",
    radicals: "Radical: 丨 (line) through 口 (mouth/enclosure)",
    exampleSentence: "我来中国了。",
    exampleBreakdown: "I (我 wǒ) come to (来 lái) middle (中 zhōng) country (国 guó)"
  },
  {
    id: "tian1", char: "天", pinyin: "tiān", meaning: "sky / day / heaven",
    category: "common",
    mnemonic: "A person (大) with a line above their head — that's the sky above! What's over the biggest person? The heavens!",
    visual: "🌤️ 一 over 大",
    soundBridge: "\"tiān\" sounds like \"tee-en\" — teens look up at the sky and dream",
    radicals: "一 (one/sky) + 大 (big/person)",
    exampleSentence: "今天天气很好。",
    exampleBreakdown: "today (今天 jīntiān) weather (天气 tiānqì) very (很 hěn) good (好 hǎo)"
  },
  {
    id: "wang2", char: "王", pinyin: "wáng", meaning: "king / ruler",
    category: "common",
    mnemonic: "Three horizontal lines connected by a vertical line — heaven (top), earth (bottom), and humankind (middle), all united by the king!",
    visual: "👑 三 + |",
    soundBridge: "\"wáng\" sounds like \"wong\" — the king has a magic wand",
    radicals: "Radical: 王/玉 (king/jade)",
    exampleSentence: "他姓王。",
    exampleBreakdown: "he (他 tā) surname (姓 xìng) king (王 wáng)"
  },
  {
    id: "ma3", char: "马", pinyin: "mǎ", meaning: "horse",
    category: "common",
    mnemonic: "The simplified form still hints at a horse seen from the side — the top is the mane, the strokes are the body, and the bottom dots are galloping hooves!",
    visual: "🐴 フ |. .",
    soundBridge: "\"mǎ\" sounds like \"ma\" — calling for mama on horseback!",
    radicals: "Radical: 马 (horse) — simplified from 馬",
    exampleSentence: "我们骑马回去。",
    exampleBreakdown: "we (我们 wǒmen) ride (骑 qí) horse (马 mǎ) go back (回去 huíqù)"
  },
  {
    id: "men2", char: "门", pinyin: "mén", meaning: "door / gate",
    category: "common",
    mnemonic: "A pair of saloon doors! The simplified character shows the frame of a doorway — two sides and a top. Walk right through!",
    visual: "🚪 | ‖",
    soundBridge: "\"mén\" sounds like \"men\" — men walk through the door",
    radicals: "Radical: 门 (door) — simplified from 門",
    exampleSentence: "我走进大门。",
    exampleBreakdown: "I (我 wǒ) walk into (走进 zǒujìn) big (大 dà) door (门 mén)"
  },
  {
    id: "li4", char: "力", pinyin: "lì", meaning: "power / strength",
    category: "common",
    mnemonic: "A flexed arm showing off a bicep! The curved stroke is the bent arm, straining with power and might.",
    visual: "💪 フ",
    soundBridge: "\"lì\" sounds like \"lee\" — Bruce Lee, the master of power!",
    radicals: "Radical: 力 (power) — this IS the radical",
    exampleSentence: "爬山需要力气。",
    exampleBreakdown: "climbing (爬山 páshān) needs (需要 xūyào) strength (力气 lìqì)"
  }
];

// ==========================================
// Stroke order hints for writing practice
// ==========================================

const STROKE_HINTS = {
  "山": "3 strokes: ① center vertical (down) ② left vertical (down) ③ right vertical (down) with connecting horizontal",
  "水": "4 strokes: ① center vertical hook ② left-falling ③ left dot ④ right-falling",
  "火": "4 strokes: ① left dot ② right dot ③ left-falling ④ right-falling with center",
  "木": "4 strokes: ① horizontal ② vertical (down) ③ left-falling ④ right-falling",
  "日": "4 strokes: ① left vertical ② top-right (corner turn) ③ inner horizontal ④ bottom closing",
  "月": "4 strokes: ① left-falling ② top-right corner turn down ③ inner horizontal ④ inner horizontal",
  "雨": "8 strokes: ① horizontal (sky) ② left vertical ③ right corner turn ④ inner vertical ⑤–⑧ four dots (rain)",
  "田": "5 strokes: ① left vertical ② top-right corner ③ inner horizontal ④ inner vertical ⑤ bottom closing",
  "石": "5 strokes: ① horizontal ② left-falling ③ left vertical of 口 ④ corner turn ⑤ bottom closing",
  "风": "4 strokes: ① left-falling ② horizontal turn down ③ left-falling inside ④ dot",
  "人": "2 strokes: ① left-falling ② right-falling — like two legs walking",
  "大": "3 strokes: ① horizontal ② left-falling ③ right-falling (dot at top optional)",
  "小": "3 strokes: ① center vertical hook ② left dot ③ right dot",
  "口": "3 strokes: ① left vertical ② top-right corner turn ③ bottom closing horizontal",
  "手": "4 strokes: ① top horizontal ② second horizontal ③ third horizontal ④ vertical hook",
  "目": "5 strokes: ① left vertical ② top-right corner ③ inner horizontal ④ inner horizontal ⑤ bottom closing",
  "女": "3 strokes: ① right-curving ② left-falling ③ horizontal",
  "子": "3 strokes: ① horizontal hook ② curved vertical ③ horizontal",
  "心": "4 strokes: ① left dot ② center-left curve ③ center dot ④ right dot",
  "一": "1 stroke: ① horizontal (left to right)",
  "二": "2 strokes: ① top horizontal (shorter) ② bottom horizontal (longer)",
  "三": "3 strokes: ① top horizontal ② middle horizontal ③ bottom horizontal",
  "四": "5 strokes: ① left vertical ② top-right corner ③ inner left-falling ④ inner curved ⑤ bottom closing",
  "五": "4 strokes: ① top horizontal ② left vertical-turn ③ inner horizontal ④ bottom horizontal",
  "好": "6 strokes: write 女 (3 strokes) on left, then 子 (3 strokes) on right",
  "中": "4 strokes: ① left vertical ② top-right corner turn ③ bottom closing ④ center vertical (piercing through)",
  "天": "4 strokes: ① top horizontal ② second horizontal ③ left-falling ④ right-falling",
  "王": "4 strokes: ① top horizontal ② middle horizontal ③ vertical ④ bottom horizontal",
  "马": "3 strokes: ① horizontal fold ② vertical hook turn ③ horizontal",
  "门": "3 strokes: ① left dot ② top-left corner turn ③ horizontal fold hook",
  "力": "2 strokes: ① horizontal fold hook ② left-falling"
};

// ==========================================
// SM-2 Spaced Repetition Algorithm
// ==========================================

const SM2 = {
  /** Default state for a new card */
  defaultState() {
    return {
      easeFactor: 2.5,
      interval: 0,
      repetitions: 0,
      nextReview: null, // null means never reviewed = new card
      known: false
    };
  },

  /**
   * Process a review rating.
   * @param {object} state - current SRS state
   * @param {number} quality - 0=Again, 1=Hard, 2=Good, 3=Easy (mapped to SM-2 0-5)
   * @returns {object} updated state
   */
  review(state, quality) {
    // Map our 0-3 scale to SM-2's 0-5 scale
    const q = [0, 2, 4, 5][quality];
    const s = { ...state };

    if (q < 3) {
      // Failed — reset
      s.repetitions = 0;
      s.interval = 0;
    } else {
      if (s.repetitions === 0) {
        s.interval = 1;
      } else if (s.repetitions === 1) {
        s.interval = 6;
      } else {
        s.interval = Math.round(s.interval * s.easeFactor);
      }
      s.repetitions += 1;
    }

    // Update ease factor
    s.easeFactor = s.easeFactor + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02));
    if (s.easeFactor < 1.3) s.easeFactor = 1.3;

    // Calculate next review date
    const now = new Date();
    if (s.interval === 0) {
      // Review again in 1 minute (for this session) — stored as today
      s.nextReview = this.todayStr();
    } else {
      const next = new Date(now.getTime() + s.interval * 86400000);
      s.nextReview = this.dateStr(next);
    }

    return s;
  },

  /** Get next interval preview for each rating */
  previewIntervals(state) {
    return [0, 1, 2, 3].map(q => {
      const result = this.review(state, q);
      return result.interval;
    });
  },

  /** Format interval as human string */
  formatInterval(days) {
    if (days === 0) return "< 1m";
    if (days === 1) return "1d";
    if (days < 30) return days + "d";
    if (days < 365) return Math.round(days / 30) + "mo";
    return (days / 365).toFixed(1) + "y";
  },

  todayStr() {
    return this.dateStr(new Date());
  },

  dateStr(d) {
    return d.toISOString().slice(0, 10);
  },

  isDue(state) {
    if (!state.nextReview) return false; // new card, not due
    return state.nextReview <= this.todayStr();
  },

  isNew(state) {
    return state.nextReview === null;
  }
};

// ==========================================
// Server Config
// ==========================================

const SERVER_URL = "http://localhost:8787";

// ==========================================
// App State & Storage
// ==========================================

const STORAGE_KEYS = {
  deck: "hanzi_deck",
  srs: "hanzi_srs",
  session: "hanzi_session",
  streak: "hanzi_streak",
  theme: "hanzi_theme",
  category: "hanzi_category"
};

const App = {
  deck: [],
  srsData: {},
  session: { date: SM2.todayStr(), reviewed: 0, correct: 0 },
  streak: { count: 0, lastDate: null },
  activeCategory: "all",
  studyQueue: [],
  currentCardIndex: 0,
  isFlipped: false,
  selectedCardId: null,
  writingCard: null,
  serverAvailable: false,
  variantData: {},       // { char: [1,2,3,4] } — available variants per character
  audioSelections: {},   // { char: { variant: N } or { allBad: true } }
  lastPlayedVariant: 0,  // last variant number played in picker
  _currentVariantAudio: null, // currently playing variant Audio object

  // --- Initialization ---

  async init() {
    this.loadData();
    this.bindEvents();
    this.updateTheme();
    this.refreshStudyView();
    this.updateDashboard();

    // Non-blocking server detection
    await this.detectServer();
  },

  loadData() {
    // Load deck
    const savedDeck = localStorage.getItem(STORAGE_KEYS.deck);
    if (savedDeck) {
      this.deck = JSON.parse(savedDeck);
    } else {
      this.deck = DEFAULT_DECK.map(c => ({ ...c }));
      this.saveDeck();
    }

    // Load SRS data
    const savedSRS = localStorage.getItem(STORAGE_KEYS.srs);
    if (savedSRS) {
      this.srsData = JSON.parse(savedSRS);
    }
    // Ensure every card has SRS data
    this.deck.forEach(card => {
      if (!this.srsData[card.id]) {
        this.srsData[card.id] = SM2.defaultState();
      }
    });
    this.saveSRS();

    // Load session
    const savedSession = localStorage.getItem(STORAGE_KEYS.session);
    if (savedSession) {
      this.session = JSON.parse(savedSession);
      if (this.session.date !== SM2.todayStr()) {
        this.session = { date: SM2.todayStr(), reviewed: 0, correct: 0 };
      }
    }

    // Load streak
    const savedStreak = localStorage.getItem(STORAGE_KEYS.streak);
    if (savedStreak) {
      this.streak = JSON.parse(savedStreak);
    }

    // Load theme
    const savedTheme = localStorage.getItem(STORAGE_KEYS.theme);
    if (savedTheme) {
      document.documentElement.setAttribute("data-theme", savedTheme);
    }

    // Load active category
    const savedCat = localStorage.getItem(STORAGE_KEYS.category);
    if (savedCat) this.activeCategory = savedCat;
  },

  saveDeck() {
    localStorage.setItem(STORAGE_KEYS.deck, JSON.stringify(this.deck));
  },

  saveSRS() {
    localStorage.setItem(STORAGE_KEYS.srs, JSON.stringify(this.srsData));
    this.syncToServer();
  },

  saveSession() {
    localStorage.setItem(STORAGE_KEYS.session, JSON.stringify(this.session));
    this.syncToServer();
  },

  saveStreak() {
    localStorage.setItem(STORAGE_KEYS.streak, JSON.stringify(this.streak));
    this.syncToServer();
  },

  // --- Server Sync ---

  async detectServer() {
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 1500);
      const resp = await fetch(`${SERVER_URL}/progress`, { signal: ctrl.signal });
      clearTimeout(timer);
      if (resp.ok) {
        this.serverAvailable = true;
        document.getElementById("server-indicator").style.display = "";
        // Load progress from server (overrides localStorage)
        const data = await resp.json();
        if (data && data.srsData) {
          this.srsData = data.srsData;
          this.session = data.session || this.session;
          this.streak = data.streak || this.streak;
          // Re-ensure every card has SRS data
          this.deck.forEach(card => {
            if (!this.srsData[card.id]) this.srsData[card.id] = SM2.defaultState();
          });
          this.refreshStudyView();
          this.updateDashboard();
        }
        // Load audio selections
        await this.loadAudioSelections();
        // Load variant info
        await this.loadVariantData();
        // Re-trigger picker on currently displayed card
        if (this.studyQueue.length > 0 && this.currentCardIndex < this.studyQueue.length) {
          this.updateAudioPicker(this.studyQueue[this.currentCardIndex]);
        }
      }
    } catch {
      // Server not available — localStorage-only mode
      this.serverAvailable = false;
    }
  },

  async syncToServer() {
    if (!this.serverAvailable) return;
    try {
      await fetch(`${SERVER_URL}/progress`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          srsData: this.srsData,
          session: this.session,
          streak: this.streak,
        }),
      });
    } catch {
      // Silently fail — localStorage still has the data
    }
  },

  async loadAudioSelections() {
    if (!this.serverAvailable) return;
    try {
      const resp = await fetch(`${SERVER_URL}/audio-selections`);
      if (resp.ok) {
        this.audioSelections = await resp.json();
      }
    } catch { /* ignore */ }
  },

  async saveAudioSelections() {
    if (!this.serverAvailable) return;
    try {
      await fetch(`${SERVER_URL}/audio-selections`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(this.audioSelections),
      });
    } catch { /* ignore */ }
  },

  async loadVariantData() {
    if (!this.serverAvailable) return;
    try {
      const resp = await fetch(`${SERVER_URL}/audio/variants`);
      if (resp.ok) {
        this.variantData = await resp.json();
      }
    } catch { /* ignore */ }
  },

  // --- Event Binding ---

  bindEvents() {
    // Navigation
    document.querySelectorAll(".nav-tab").forEach(tab => {
      tab.addEventListener("click", () => this.switchView(tab.dataset.view));
    });

    // Theme toggle
    document.getElementById("btn-theme").addEventListener("click", () => this.toggleTheme());

    // Search
    document.getElementById("btn-search").addEventListener("click", () => this.openSearch());
    document.getElementById("search-close").addEventListener("click", () => this.closeSearch());
    document.getElementById("search-input").addEventListener("input", (e) => this.handleSearch(e.target.value));

    // Category filters
    document.querySelectorAll(".cat-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        this.activeCategory = btn.dataset.cat;
        localStorage.setItem(STORAGE_KEYS.category, this.activeCategory);
        document.querySelectorAll(".cat-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        this.refreshStudyView();
      });
    });

    // Flashcard
    document.getElementById("flashcard").addEventListener("click", (e) => {
      // Don't flip if clicking audio button
      if (e.target.closest(".btn-audio")) return;
      this.flipCard();
    });

    // Rating buttons
    document.querySelectorAll(".btn-rate").forEach(btn => {
      btn.addEventListener("click", () => this.rateCard(parseInt(btn.dataset.rating)));
    });

    // Audio (back of card only)
    document.getElementById("btn-audio").addEventListener("click", (e) => {
      e.stopPropagation();
      const card = this.studyQueue[this.currentCardIndex];
      if (card) this.speak(card.char);
    });

    // Audio picker — variant buttons
    document.querySelectorAll(".btn-variant").forEach(btn => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const card = this.studyQueue[this.currentCardIndex];
        if (card) this.playVariant(card.char, parseInt(btn.dataset.variant));
      });
    });

    // Audio picker — Keep button
    document.getElementById("btn-audio-keep").addEventListener("click", (e) => {
      e.stopPropagation();
      const card = this.studyQueue[this.currentCardIndex];
      if (card && this.lastPlayedVariant > 0) this.promoteVariant(card.char, this.lastPlayedVariant);
    });

    // Audio picker — All Bad button
    document.getElementById("btn-audio-bad").addEventListener("click", (e) => {
      e.stopPropagation();
      const card = this.studyQueue[this.currentCardIndex];
      if (card) this.markAllBad(card.char);
    });

    // Skip
    document.getElementById("btn-skip").addEventListener("click", () => this.skipCard());

    // Learn new
    document.getElementById("btn-learn-new").addEventListener("click", () => this.learnNewCards());

    // Browse
    document.getElementById("browse-filter").addEventListener("change", () => this.renderBrowse());
    document.getElementById("browse-sort").addEventListener("change", () => this.renderBrowse());
    document.getElementById("detail-close").addEventListener("click", () => this.closeDetail());
    document.getElementById("btn-toggle-known").addEventListener("click", () => this.toggleKnown());
    document.getElementById("btn-practice-write").addEventListener("click", () => {
      this.closeDetail();
      this.openWritingPractice(this.selectedCardId);
    });
    document.getElementById("btn-detail-audio").addEventListener("click", () => {
      const card = this.deck.find(c => c.id === this.selectedCardId);
      if (card) this.speak(card.char);
    });

    // Writing practice
    this.initCanvas();
    document.getElementById("btn-clear-canvas").addEventListener("click", () => this.clearCanvas());
    document.getElementById("btn-show-strokes").addEventListener("click", () => this.showStrokeHints());
    document.getElementById("btn-close-writing").addEventListener("click", () => this.closeWritingPractice());

    // Manage - Add card
    document.getElementById("add-card-form").addEventListener("submit", (e) => {
      e.preventDefault();
      this.addCustomCard();
    });

    // Manage - Import/Export
    document.getElementById("btn-export").addEventListener("click", () => this.exportDeck());
    document.getElementById("btn-import").addEventListener("click", () => {
      document.getElementById("import-file").click();
    });
    document.getElementById("import-file").addEventListener("change", (e) => this.importDeck(e));

    // Manage - Reset
    document.getElementById("btn-reset-progress").addEventListener("click", () => this.resetProgress());
    document.getElementById("btn-reset-all").addEventListener("click", () => this.resetAll());

    // Keyboard shortcuts
    document.addEventListener("keydown", (e) => this.handleKeyboard(e));

    // Close overlays on background click
    document.getElementById("search-overlay").addEventListener("click", (e) => {
      if (e.target === e.currentTarget) this.closeSearch();
    });
    document.getElementById("card-detail-modal").addEventListener("click", (e) => {
      if (e.target === e.currentTarget) this.closeDetail();
    });

    // Restore active category button
    if (this.activeCategory !== "all") {
      document.querySelectorAll(".cat-btn").forEach(b => {
        b.classList.toggle("active", b.dataset.cat === this.activeCategory);
      });
    }
  },

  handleKeyboard(e) {
    // Only handle shortcuts when not typing in input
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;

    const view = document.querySelector(".view.active");
    if (!view || view.id !== "view-study") return;

    if (e.key === " " || e.key === "Enter") {
      e.preventDefault();
      if (!this.isFlipped) {
        this.flipCard();
      }
    } else if (this.isFlipped) {
      if (e.key === "1") this.rateCard(0);
      else if (e.key === "2") this.rateCard(1);
      else if (e.key === "3") this.rateCard(2);
      else if (e.key === "4") this.rateCard(3);
    }
  },

  // --- View Switching ---

  switchView(viewName) {
    document.querySelectorAll(".nav-tab").forEach(t => t.classList.remove("active"));
    document.querySelector(`.nav-tab[data-view="${viewName}"]`).classList.add("active");
    document.querySelectorAll(".view").forEach(v => {
      v.classList.remove("active");
      v.classList.add("hidden");
    });
    const view = document.getElementById(`view-${viewName}`);
    view.classList.remove("hidden");
    view.classList.add("active");

    if (viewName === "browse") this.renderBrowse();
    if (viewName === "stats") this.renderStats();
  },

  // --- Theme ---

  toggleTheme() {
    const current = document.documentElement.getAttribute("data-theme");
    const next = current === "light" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", next === "dark" ? "" : "light");
    localStorage.setItem(STORAGE_KEYS.theme, next === "dark" ? "" : "light");
    this.updateTheme();
  },

  updateTheme() {
    const isLight = document.documentElement.getAttribute("data-theme") === "light";
    document.getElementById("btn-theme").textContent = isLight ? "☀️" : "🌙";
  },

  // --- Study Logic ---

  refreshStudyView() {
    const filteredDeck = this.getFilteredDeck();
    const due = [];
    const newCards = [];

    filteredDeck.forEach(card => {
      const srs = this.srsData[card.id];
      if (srs.known) return;
      if (SM2.isDue(srs)) due.push(card);
      else if (SM2.isNew(srs)) newCards.push(card);
    });

    // Study queue: due cards first, then up to 5 new cards
    this.studyQueue = [...due, ...newCards.slice(0, 5)];
    this.currentCardIndex = 0;
    this.isFlipped = false;

    const container = document.getElementById("flashcard-container");
    const empty = document.getElementById("study-empty");
    const writing = document.getElementById("writing-practice");

    writing.classList.add("hidden");

    if (this.studyQueue.length === 0) {
      container.classList.add("hidden");
      empty.classList.remove("hidden");
    } else {
      container.classList.remove("hidden");
      empty.classList.add("hidden");
      this.showCard();
    }

    this.updateDashboard();
  },

  getFilteredDeck() {
    if (this.activeCategory === "all") return this.deck;
    return this.deck.filter(c => c.category === this.activeCategory);
  },

  showCard() {
    if (this.currentCardIndex >= this.studyQueue.length) {
      this.refreshStudyView();
      return;
    }

    const card = this.studyQueue[this.currentCardIndex];
    const srs = this.srsData[card.id];

    // Front
    document.getElementById("card-char").textContent = card.char;
    document.getElementById("card-pinyin").textContent = card.pinyin;
    document.getElementById("card-cat-tag").textContent = card.category;

    // Back
    document.getElementById("card-char-back").textContent = card.char;
    document.getElementById("card-pinyin-back").textContent = card.pinyin;
    document.getElementById("card-meaning").textContent = card.meaning;
    document.getElementById("card-mnemonic").textContent = card.mnemonic || "";
    document.getElementById("card-visual").textContent = card.visual || "";
    document.getElementById("card-sound-bridge").textContent = card.soundBridge || "";
    document.getElementById("card-radicals").textContent = card.radicals || "";
    document.getElementById("card-example-sentence").textContent = card.exampleSentence || "";
    document.getElementById("card-example-breakdown").textContent = card.exampleBreakdown || "";

    // Progress
    document.getElementById("card-progress-text").textContent =
      `${this.currentCardIndex + 1} / ${this.studyQueue.length}`;

    // Rating time previews
    const intervals = SM2.previewIntervals(srs);
    document.getElementById("rate-again-time").textContent = SM2.formatInterval(intervals[0]);
    document.getElementById("rate-hard-time").textContent = SM2.formatInterval(intervals[1]);
    document.getElementById("rate-good-time").textContent = SM2.formatInterval(intervals[2]);
    document.getElementById("rate-easy-time").textContent = SM2.formatInterval(intervals[3]);

    // Audio picker — show variant buttons if variants exist
    this.updateAudioPicker(card);

    // Reset flip state
    this.isFlipped = false;
    document.getElementById("flashcard").classList.remove("flipped");
    document.getElementById("rating-buttons").classList.add("hidden");
  },

  flipCard() {
    if (this.studyQueue.length === 0) return;
    this.isFlipped = !this.isFlipped;
    document.getElementById("flashcard").classList.toggle("flipped", this.isFlipped);
    document.getElementById("rating-buttons").classList.toggle("hidden", !this.isFlipped);
    this.resizeCardInner();
  },

  resizeCardInner() {
    const inner = document.querySelector(".card-inner");
    if (this.isFlipped) {
      const back = document.querySelector(".card-back");
      // Temporarily measure back content height
      requestAnimationFrame(() => {
        const h = back.scrollHeight;
        inner.style.minHeight = Math.max(400, h) + "px";
      });
    } else {
      inner.style.minHeight = "400px";
    }
  },

  rateCard(quality) {
    const card = this.studyQueue[this.currentCardIndex];
    if (!card) return;

    const srs = this.srsData[card.id];
    this.srsData[card.id] = SM2.review(srs, quality);
    this.saveSRS();

    // Update session
    this.session.reviewed++;
    if (quality >= 2) this.session.correct++;
    this.saveSession();

    // Update streak
    this.updateStreak();

    // Next card
    this.currentCardIndex++;
    if (this.currentCardIndex >= this.studyQueue.length) {
      this.refreshStudyView();
    } else {
      this.showCard();
    }

    this.updateDashboard();
  },

  skipCard() {
    this.currentCardIndex++;
    if (this.currentCardIndex >= this.studyQueue.length) {
      this.refreshStudyView();
    } else {
      this.showCard();
    }
  },

  learnNewCards() {
    const filteredDeck = this.getFilteredDeck();
    const newCards = filteredDeck.filter(c => {
      const srs = this.srsData[c.id];
      return !srs.known && SM2.isNew(srs);
    });

    if (newCards.length === 0) {
      this.toast("No new cards available in this category");
      return;
    }

    this.studyQueue = newCards.slice(0, 10);
    this.currentCardIndex = 0;
    document.getElementById("flashcard-container").classList.remove("hidden");
    document.getElementById("study-empty").classList.add("hidden");
    this.showCard();
  },

  updateStreak() {
    const today = SM2.todayStr();
    const yesterday = SM2.dateStr(new Date(Date.now() - 86400000));

    if (this.streak.lastDate === today) {
      // Already counted today
    } else if (this.streak.lastDate === yesterday) {
      this.streak.count++;
      this.streak.lastDate = today;
    } else if (!this.streak.lastDate) {
      this.streak.count = 1;
      this.streak.lastDate = today;
    } else {
      // Streak broken
      this.streak.count = 1;
      this.streak.lastDate = today;
    }
    this.saveStreak();
  },

  // --- Dashboard ---

  updateDashboard() {
    const filteredDeck = this.getFilteredDeck();
    let dueCount = 0, newCount = 0, learnedCount = 0;

    filteredDeck.forEach(card => {
      const srs = this.srsData[card.id];
      if (srs.known) return;
      if (SM2.isDue(srs)) dueCount++;
      else if (SM2.isNew(srs)) newCount++;
      else learnedCount++;
    });

    // Add known count to learned for display
    const knownCount = filteredDeck.filter(c => this.srsData[c.id].known).length;
    learnedCount += knownCount;

    document.getElementById("stat-due").textContent = dueCount;
    document.getElementById("stat-new").textContent = newCount;
    document.getElementById("stat-learned").textContent = learnedCount;
    document.getElementById("stat-streak").textContent = this.streak.count;
    document.getElementById("stat-today-count").textContent = this.session.reviewed;
    document.getElementById("stat-today-acc").textContent =
      this.session.reviewed > 0
        ? Math.round((this.session.correct / this.session.reviewed) * 100) + "%"
        : "—";
  },

  // --- Search ---

  openSearch() {
    document.getElementById("search-overlay").classList.remove("hidden");
    document.getElementById("search-input").value = "";
    document.getElementById("search-input").focus();
    document.getElementById("search-results").innerHTML = "";
  },

  closeSearch() {
    document.getElementById("search-overlay").classList.add("hidden");
  },

  handleSearch(query) {
    const q = query.toLowerCase().trim();
    const results = document.getElementById("search-results");

    if (!q) {
      results.innerHTML = "";
      return;
    }

    const matches = this.deck.filter(card =>
      card.char.includes(q) ||
      card.pinyin.toLowerCase().includes(q) ||
      card.meaning.toLowerCase().includes(q)
    );

    results.innerHTML = matches.map(card => `
      <div class="search-result-item" data-id="${card.id}">
        <div class="sr-char">${card.char}</div>
        <div class="sr-info">
          <div class="sr-pinyin">${card.pinyin}</div>
          <div class="sr-meaning">${card.meaning}</div>
        </div>
      </div>
    `).join("");

    results.querySelectorAll(".search-result-item").forEach(el => {
      el.addEventListener("click", () => {
        this.closeSearch();
        this.openDetail(el.dataset.id);
      });
    });
  },

  // --- Browse View ---

  renderBrowse() {
    const filter = document.getElementById("browse-filter").value;
    const sort = document.getElementById("browse-sort").value;

    let cards = filter === "all" ? [...this.deck] : this.deck.filter(c => c.category === filter);

    if (sort === "character") {
      cards.sort((a, b) => a.char.localeCompare(b.char, "zh"));
    } else if (sort === "due") {
      cards.sort((a, b) => {
        const sa = this.srsData[a.id];
        const sb = this.srsData[b.id];
        const da = sa.nextReview || "9999";
        const db = sb.nextReview || "9999";
        return da.localeCompare(db);
      });
    }

    const grid = document.getElementById("browse-grid");
    grid.innerHTML = cards.map(card => {
      const srs = this.srsData[card.id];
      let badge = "";
      if (srs.known) badge = '<span class="bc-badge learned">known</span>';
      else if (SM2.isDue(srs)) badge = '<span class="bc-badge due">due</span>';
      else if (SM2.isNew(srs)) badge = '<span class="bc-badge new">new</span>';
      else badge = '<span class="bc-badge learned">learned</span>';

      return `
        <div class="browse-card ${srs.known ? 'known' : ''}" data-id="${card.id}">
          ${badge}
          <div class="bc-char">${card.char}</div>
          <div class="bc-pinyin">${card.pinyin}</div>
          <div class="bc-meaning">${card.meaning}</div>
        </div>
      `;
    }).join("");

    grid.querySelectorAll(".browse-card").forEach(el => {
      el.addEventListener("click", () => this.openDetail(el.dataset.id));
    });
  },

  // --- Card Detail ---

  openDetail(cardId) {
    this.selectedCardId = cardId;
    const card = this.deck.find(c => c.id === cardId);
    if (!card) return;

    const srs = this.srsData[cardId];
    const content = document.getElementById("detail-content");

    content.innerHTML = `
      <div class="detail-char">${card.char}</div>
      <div class="detail-pinyin">${card.pinyin}</div>
      <div class="detail-meaning">${card.meaning}</div>
      ${card.visual ? `<div class="detail-visual">${card.visual}</div>` : ""}
      ${card.mnemonic ? `<div class="detail-mnemonic">"${card.mnemonic}"</div>` : ""}
      ${card.soundBridge ? `<div class="detail-sound-bridge">${card.soundBridge}</div>` : ""}
      ${card.radicals ? `<div class="detail-radicals">${card.radicals}</div>` : ""}
    `;

    document.getElementById("btn-toggle-known").textContent =
      srs.known ? "Unmark Known" : "Mark as Known";

    document.getElementById("card-detail-modal").classList.remove("hidden");
  },

  closeDetail() {
    document.getElementById("card-detail-modal").classList.add("hidden");
    this.selectedCardId = null;
  },

  toggleKnown() {
    if (!this.selectedCardId) return;
    const srs = this.srsData[this.selectedCardId];
    srs.known = !srs.known;
    this.saveSRS();

    document.getElementById("btn-toggle-known").textContent =
      srs.known ? "Unmark Known" : "Mark as Known";

    this.toast(srs.known ? "Marked as known — will be skipped" : "Unmarked — will appear in reviews");
    this.updateDashboard();
    this.renderBrowse();
  },

  // --- Audio (Qwen3 TTS pre-generated) ---

  speak(text) {
    // Use pre-generated Qwen3 TTS audio files (Serena voice)
    const char = text.trim();
    const audioPath = `audio/${encodeURIComponent(char)}.mp3`;
    const audio = new Audio(audioPath);
    audio.onerror = () => {
      // Fallback to Web Speech API if audio file not found
      if (!("speechSynthesis" in window)) {
        this.toast("Audio not available");
        return;
      }
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = "zh-CN";
      utterance.rate = 0.8;
      const voices = speechSynthesis.getVoices();
      const zhVoice = voices.find(v => v.lang.startsWith("zh"));
      if (zhVoice) utterance.voice = zhVoice;
      speechSynthesis.cancel();
      speechSynthesis.speak(utterance);
    };
    audio.play();
  },

  // --- Audio Picker (variant curation) ---

  /** Check if a card has variants and show/hide picker accordingly */
  updateAudioPicker(card) {
    const picker = document.getElementById("audio-picker");
    const mainBtn = document.getElementById("btn-audio");
    const variants = this.variantData[card.char];

    if (this.serverAvailable && variants && variants.length > 0) {
      // Show picker, hide single listen button
      picker.classList.remove("hidden");
      mainBtn.classList.add("hidden");
      this.lastPlayedVariant = 0;
      // Reset variant button states
      document.querySelectorAll(".btn-variant").forEach(btn => {
        btn.classList.remove("playing");
        const v = parseInt(btn.dataset.variant);
        btn.disabled = !variants.includes(v);
        btn.style.opacity = variants.includes(v) ? "1" : "0.3";
      });
    } else {
      // Normal single audio mode
      picker.classList.add("hidden");
      mainBtn.classList.remove("hidden");
    }
  },

  playVariant(char, variant) {
    // Stop any currently playing variant
    if (this._currentVariantAudio) {
      this._currentVariantAudio.pause();
      this._currentVariantAudio = null;
    }

    const url = `${SERVER_URL}/audio/variants/${encodeURIComponent(char)}_v${variant}.mp3`;
    const audio = new Audio(url);
    this._currentVariantAudio = audio;
    this.lastPlayedVariant = variant;

    // Update button states
    document.querySelectorAll(".btn-variant").forEach(btn => {
      btn.classList.toggle("playing", parseInt(btn.dataset.variant) === variant);
    });

    audio.play().catch(() => this.toast("Failed to play variant"));
  },

  async promoteVariant(char, variant) {
    if (!this.serverAvailable) return;
    try {
      const resp = await fetch(`${SERVER_URL}/audio/promote`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ char, variant }),
      });
      if (resp.ok) {
        // Update local state — remove variants for this char
        delete this.variantData[char];
        this.audioSelections[char] = { variant, promoted: true };
        await this.saveAudioSelections();
        this.toast(`Kept variant ${variant} for ${char}`);
        // Refresh picker UI for current card
        const card = this.studyQueue[this.currentCardIndex];
        if (card) this.updateAudioPicker(card);
      }
    } catch {
      this.toast("Failed to promote variant");
    }
  },

  async markAllBad(char) {
    this.audioSelections[char] = { allBad: true };
    await this.saveAudioSelections();
    this.toast(`Marked all variants for ${char} as bad`);
  },

  // --- Writing Practice ---

  initCanvas() {
    const canvas = document.getElementById("writing-canvas");
    const ctx = canvas.getContext("2d");
    let drawing = false;
    let lastX = 0, lastY = 0;

    const getPos = (e) => {
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      if (e.touches) {
        return {
          x: (e.touches[0].clientX - rect.left) * scaleX,
          y: (e.touches[0].clientY - rect.top) * scaleY
        };
      }
      return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
      };
    };

    const start = (e) => {
      e.preventDefault();
      drawing = true;
      const pos = getPos(e);
      lastX = pos.x;
      lastY = pos.y;
    };

    const move = (e) => {
      if (!drawing) return;
      e.preventDefault();
      const pos = getPos(e);
      ctx.beginPath();
      ctx.moveTo(lastX, lastY);
      ctx.lineTo(pos.x, pos.y);
      ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();
      ctx.lineWidth = 4;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.stroke();
      lastX = pos.x;
      lastY = pos.y;
    };

    const end = () => { drawing = false; };

    canvas.addEventListener("mousedown", start);
    canvas.addEventListener("mousemove", move);
    canvas.addEventListener("mouseup", end);
    canvas.addEventListener("mouseleave", end);
    canvas.addEventListener("touchstart", start, { passive: false });
    canvas.addEventListener("touchmove", move, { passive: false });
    canvas.addEventListener("touchend", end);
  },

  clearCanvas() {
    const canvas = document.getElementById("writing-canvas");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw grid guide
    this.drawCanvasGrid(ctx, canvas);
  },

  drawCanvasGrid(ctx, canvas) {
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--border").trim();
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);

    // Cross guide lines
    ctx.beginPath();
    ctx.moveTo(canvas.width / 2, 0);
    ctx.lineTo(canvas.width / 2, canvas.height);
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();

    // Diagonal guides
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(canvas.width, canvas.height);
    ctx.moveTo(canvas.width, 0);
    ctx.lineTo(0, canvas.height);
    ctx.stroke();

    ctx.setLineDash([]);
  },

  openWritingPractice(cardId) {
    const card = this.deck.find(c => c.id === cardId);
    if (!card) return;
    this.writingCard = card;

    const wp = document.getElementById("writing-practice");
    wp.classList.remove("hidden");
    document.getElementById("stroke-hints").classList.add("hidden");
    this.clearCanvas();

    this.switchView("study");
  },

  showStrokeHints() {
    if (!this.writingCard) return;
    const hints = document.getElementById("stroke-hints");
    const charHints = STROKE_HINTS[this.writingCard.char];
    hints.textContent = charHints || `No stroke hints for ${this.writingCard.char}`;
    hints.classList.remove("hidden");
  },

  closeWritingPractice() {
    document.getElementById("writing-practice").classList.add("hidden");
    this.writingCard = null;
  },

  // --- Stats View ---

  renderStats() {
    const total = this.deck.length;
    let learned = 0, known = 0, due = 0;

    this.deck.forEach(card => {
      const srs = this.srsData[card.id];
      if (srs.known) { known++; learned++; }
      else if (SM2.isDue(srs)) { due++; learned++; }
      else if (!SM2.isNew(srs)) { learned++; }
    });

    document.getElementById("stats-total").textContent = total;
    document.getElementById("stats-learned").textContent = learned;
    document.getElementById("stats-known").textContent = known;
    document.getElementById("stats-due").textContent = due;
    document.getElementById("stats-streak").textContent = this.streak.count + " days";
    document.getElementById("stats-session-count").textContent = this.session.reviewed;
    document.getElementById("stats-session-correct").textContent = this.session.correct;
    document.getElementById("stats-session-acc").textContent =
      this.session.reviewed > 0
        ? Math.round((this.session.correct / this.session.reviewed) * 100) + "%"
        : "—";

    // Forecast chart (next 7 days)
    this.renderForecast();

    // Category progress
    this.renderCategoryProgress();
  },

  renderForecast() {
    const chart = document.getElementById("forecast-chart");
    const days = [];
    const today = new Date();
    const dayNames = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

    for (let i = 0; i < 7; i++) {
      const d = new Date(today.getTime() + i * 86400000);
      const dateStr = SM2.dateStr(d);
      let count = 0;
      this.deck.forEach(card => {
        const srs = this.srsData[card.id];
        if (!srs.known && srs.nextReview === dateStr) count++;
      });
      days.push({ label: i === 0 ? "Today" : dayNames[d.getDay()], count });
    }

    const maxCount = Math.max(...days.map(d => d.count), 1);

    chart.innerHTML = days.map(d => {
      const height = Math.max(4, (d.count / maxCount) * 80);
      return `
        <div class="forecast-bar" style="height: ${height}px" title="${d.count} cards">
          <span class="bar-count">${d.count}</span>
          <span class="bar-label">${d.label}</span>
        </div>
      `;
    }).join("");
  },

  renderCategoryProgress() {
    const container = document.getElementById("category-progress");
    const categories = ["nature", "people", "numbers", "common", "custom"];
    const labels = { nature: "Nature", people: "People", numbers: "Numbers", common: "Common", custom: "Custom" };

    container.innerHTML = categories.map(cat => {
      const cards = this.deck.filter(c => c.category === cat);
      if (cards.length === 0) return "";
      const learned = cards.filter(c => {
        const srs = this.srsData[c.id];
        return srs.known || (!SM2.isNew(srs));
      }).length;
      const pct = Math.round((learned / cards.length) * 100);

      return `
        <div class="cat-progress-row">
          <span class="cat-progress-label">${labels[cat]}</span>
          <div class="cat-progress-bar">
            <div class="cat-progress-fill" style="width: ${pct}%"></div>
          </div>
          <span class="cat-progress-text">${pct}%</span>
        </div>
      `;
    }).join("");
  },

  // --- Manage: Add Custom Card ---

  addCustomCard() {
    const char = document.getElementById("new-char").value.trim();
    const pinyin = document.getElementById("new-pinyin").value.trim();
    const meaning = document.getElementById("new-meaning").value.trim();
    const mnemonic = document.getElementById("new-mnemonic").value.trim();
    const visual = document.getElementById("new-visual").value.trim();

    if (!char || !pinyin || !meaning) return;

    // Generate unique id
    const id = "custom_" + Date.now();

    const card = {
      id, char, pinyin, meaning,
      category: "custom",
      mnemonic: mnemonic || "",
      visual: visual || "",
      soundBridge: "",
      radicals: ""
    };

    this.deck.push(card);
    this.srsData[id] = SM2.defaultState();
    this.saveDeck();
    this.saveSRS();

    // Reset form
    document.getElementById("add-card-form").reset();
    this.toast(`Added "${char}" to your deck!`);
    this.updateDashboard();
  },

  // --- Import / Export ---

  exportDeck() {
    const data = {
      version: 1,
      deck: this.deck,
      srs: this.srsData,
      exportDate: new Date().toISOString()
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `hanzi-deck-${SM2.todayStr()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    this.toast("Deck exported!");
  },

  importDeck(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target.result);
        if (data.deck && Array.isArray(data.deck)) {
          // Merge imported cards
          let added = 0;
          data.deck.forEach(card => {
            if (!this.deck.find(c => c.id === card.id)) {
              this.deck.push(card);
              this.srsData[card.id] = (data.srs && data.srs[card.id]) || SM2.defaultState();
              added++;
            }
          });
          this.saveDeck();
          this.saveSRS();
          this.toast(`Imported ${added} new cards!`);
          this.updateDashboard();
          this.refreshStudyView();
        } else {
          this.toast("Invalid deck file format");
        }
      } catch {
        this.toast("Error reading file");
      }
    };
    reader.readAsText(file);
    event.target.value = "";
  },

  // --- Reset ---

  resetProgress() {
    if (!confirm("Reset all SRS progress? Your cards will be kept but all review history will be erased.")) return;

    this.deck.forEach(card => {
      this.srsData[card.id] = SM2.defaultState();
    });
    this.session = { date: SM2.todayStr(), reviewed: 0, correct: 0 };
    this.streak = { count: 0, lastDate: null };
    this.saveSRS();
    this.saveSession();
    this.saveStreak();
    this.refreshStudyView();
    this.toast("Progress reset!");
  },

  resetAll() {
    if (!confirm("Reset EVERYTHING? This will delete all custom cards and progress.")) return;

    localStorage.removeItem(STORAGE_KEYS.deck);
    localStorage.removeItem(STORAGE_KEYS.srs);
    localStorage.removeItem(STORAGE_KEYS.session);
    localStorage.removeItem(STORAGE_KEYS.streak);
    this.deck = DEFAULT_DECK.map(c => ({ ...c }));
    this.srsData = {};
    this.deck.forEach(card => {
      this.srsData[card.id] = SM2.defaultState();
    });
    this.session = { date: SM2.todayStr(), reviewed: 0, correct: 0 };
    this.streak = { count: 0, lastDate: null };
    this.saveDeck();
    this.saveSRS();
    this.saveSession();
    this.saveStreak();
    this.refreshStudyView();
    this.toast("Everything reset to defaults!");
  },

  // --- Toast ---

  toast(message) {
    const el = document.getElementById("toast");
    el.textContent = message;
    el.classList.remove("hidden");
    clearTimeout(this._toastTimer);
    this._toastTimer = setTimeout(() => el.classList.add("hidden"), 2500);
  }
};

// --- Boot ---
document.addEventListener("DOMContentLoaded", () => {
  // Preload voices for speech synthesis
  if ("speechSynthesis" in window) {
    speechSynthesis.getVoices();
    speechSynthesis.addEventListener("voiceschanged", () => speechSynthesis.getVoices());
  }
  App.init();
});
