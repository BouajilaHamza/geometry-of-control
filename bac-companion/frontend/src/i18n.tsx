import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'

export type Lang = 'fr' | 'ar'

type Dict = Record<string, { fr: string; ar: string }>

// All UI copy. Tone is deliberately supportive — see the PRD's "Tone and
// language" section. We avoid "late / missed / overdue" in favour of recovery
// framing.
const STRINGS: Dict = {
  appName: { fr: 'Bac Companion', ar: 'رفيق الباكالوريا' },
  navToday: { fr: "Aujourd'hui", ar: 'اليوم' },
  navWeek: { fr: 'Semaine', ar: 'الأسبوع' },
  navMastery: { fr: 'Progrès', ar: 'التقدّم' },
  navProfile: { fr: 'Profil', ar: 'الحساب' },

  greetingMorning: { fr: 'Bonjour', ar: 'صباح الخير' },
  todayQuestion: { fr: "À faire aujourd'hui", ar: 'مهام اليوم' },
  weakQuestion: { fr: 'Points à renforcer', ar: 'نقاط للتحسين' },
  trackQuestion: { fr: 'Cette semaine', ar: 'هذا الأسبوع' },

  startSession: { fr: 'Commencer la séance', ar: 'ابدأ الجلسة' },
  continueSession: { fr: 'Continuer', ar: 'واصل' },
  minutes: { fr: 'min', ar: 'دقيقة' },
  tasks: { fr: 'tâches', ar: 'مهام' },
  task: { fr: 'tâche', ar: 'مهمة' },
  streak: { fr: 'jours de suite', ar: 'أيام متتالية' },
  day: { fr: 'jour', ar: 'يوم' },

  review: { fr: 'Révision', ar: 'مراجعة' },
  drill: { fr: 'Point faible', ar: 'نقطة ضعف' },
  new: { fr: 'Nouveau', ar: 'جديد' },
  memory: { fr: 'Mémorisation', ar: 'حفظ' },
  problem: { fr: 'Exercice', ar: 'تمرين' },

  // Recovery
  recoveryTitle: { fr: 'Content de te revoir 👋', ar: 'فرحانين بيك 👋' },
  lightRestart: { fr: 'Reprendre en douceur', ar: 'نعاودوا بالشوية' },
  hiddenSetAside: { fr: 'mis de côté pour plus tard', ar: 'محطوطين على جنب' },
  startRecovery: { fr: 'Démarrer le plan léger', ar: 'ابدأ البرنامج الخفيف' },

  // Session
  howWasIt: { fr: "Comment c'était ?", ar: 'كيفاش كانت؟' },
  again: { fr: 'À revoir', ar: 'نعاودها' },
  hard: { fr: 'Difficile', ar: 'صعيبة' },
  good: { fr: 'Bien', ar: 'مليح' },
  easy: { fr: 'Facile', ar: 'ساهلة' },
  sessionDone: { fr: 'Séance terminée !', ar: 'كملت الجلسة!' },
  correctOf: { fr: 'réussies sur', ar: 'صحيحة من' },
  masteryUp: { fr: 'Maîtrise', ar: 'الإتقان' },
  backHome: { fr: "Retour à l'accueil", ar: 'للرئيسية' },
  nextTask: { fr: 'Suivant', ar: 'التالي' },

  // Mastery
  bySubject: { fr: 'Par matière', ar: 'حسب المادة' },
  weakPointsTitle: { fr: 'Tes points faibles', ar: 'نقاط الضعف متاعك' },
  noWeakPoints: { fr: 'Aucun point faible majeur — continue ! 🎉', ar: 'ما عندكش نقاط ضعف كبيرة — واصل! 🎉' },
  concepts: { fr: 'concepts', ar: 'مفاهيم' },
  errorRate: { fr: "taux d'erreur", ar: 'نسبة الخطأ' },

  // Profile / dev
  profile: { fr: 'Profil', ar: 'الحساب' },
  section: { fr: 'Section', ar: 'الشعبة' },
  language: { fr: 'Langue', ar: 'اللغة' },
  demoTools: { fr: 'Outils de démo', ar: 'أدوات العرض' },
  demoHint: {
    fr: "Simule une absence pour voir le mode récupération, puis recharge l'accueil.",
    ar: 'حاكي غياب باش تشوف وضعية الاسترجاع، ثم اعمل تحديث للرئيسية.',
  },
  simulate: { fr: 'Simuler', ar: 'حاكي' },
  daysAway: { fr: "jours d'absence", ar: 'أيام غياب' },
  reset: { fr: 'Réinitialiser la démo', ar: 'إعادة ضبط العرض' },

  onTrack: { fr: 'Sur la bonne voie', ar: 'على الطريق الصحيح' },
  mildBacklog: { fr: 'Léger retard', ar: 'تأخّر بسيط' },
  majorBacklog: { fr: 'Mode récupération', ar: 'وضعية الاسترجاع' },
  reEntry: { fr: 'Nouveau départ', ar: 'بداية جديدة' },

  loading: { fr: 'Chargement…', ar: 'جارٍ التحميل…' },
  allCaughtUp: { fr: "Rien d'urgent aujourd'hui — profite ! ✨", ar: 'ما فماش حاجة مستعجلة اليوم — ارتاح! ✨' },
  weeklyPlanTitle: { fr: 'Ton plan de la semaine', ar: 'برنامجك للأسبوع' },
}

interface LangCtx {
  lang: Lang
  dir: 'ltr' | 'rtl'
  setLang: (l: Lang) => void
  t: (key: keyof typeof STRINGS) => string
  pick: (fr: string, ar: string) => string
}

const Ctx = createContext<LangCtx | null>(null)

export function LangProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Lang>(() => (localStorage.getItem('lang') as Lang) || 'fr')
  const dir = lang === 'ar' ? 'rtl' : 'ltr'

  useEffect(() => {
    localStorage.setItem('lang', lang)
    document.documentElement.lang = lang
    document.documentElement.dir = dir
  }, [lang, dir])

  const t = (key: keyof typeof STRINGS) => STRINGS[key]?.[lang] ?? String(key)
  const pick = (fr: string, ar: string) => (lang === 'ar' ? ar : fr)

  return <Ctx.Provider value={{ lang, dir, setLang, t, pick }}>{children}</Ctx.Provider>
}

export function useLang() {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useLang must be used within LangProvider')
  return ctx
}
