import { NavLink, useLocation } from 'react-router-dom'
import { useLang } from '../i18n'

const navItems = [
  { to: '/', key: 'navToday', icon: HomeIcon },
  { to: '/week', key: 'navWeek', icon: CalendarIcon },
  { to: '/mastery', key: 'navMastery', icon: ChartIcon },
  { to: '/profile', key: 'navProfile', icon: UserIcon },
] as const

export default function Layout({ children }: { children: React.ReactNode }) {
  const { t, lang, setLang } = useLang()
  const loc = useLocation()
  const inSession = loc.pathname.startsWith('/session') || loc.pathname.startsWith('/recovery')

  return (
    <div className="mx-auto flex min-h-screen max-w-md flex-col bg-slate-50">
      {/* Top bar */}
      {!inSession && (
        <header className="sticky top-0 z-20 flex h-14 items-center justify-between bg-white/80 px-5 shadow-[0_1px_0_0_#e2e8f0] backdrop-blur">
          <div className="flex items-center gap-2.5">
            <div className="grid h-8 w-8 place-items-center rounded-xl bg-brand-600 text-white shadow-lift">
              <BookIcon />
            </div>
            <span className="text-[17px] font-extrabold tracking-tight text-ink-900">{t('appName')}</span>
          </div>
          <button
            onClick={() => setLang(lang === 'fr' ? 'ar' : 'fr')}
            className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-bold text-ink-600 transition hover:bg-slate-50"
            aria-label="toggle language"
          >
            {lang === 'fr' ? 'عربي' : 'FR'}
          </button>
        </header>
      )}

      <main className={`flex-1 px-4 pb-28 ${inSession ? 'pt-4' : 'pt-4'}`}>{children}</main>

      {/* Bottom navigation */}
      {!inSession && (
        <nav className="fixed inset-x-0 bottom-0 z-20 mx-auto max-w-md bg-white/95 shadow-[0_-1px_0_0_#e2e8f0] backdrop-blur">
          <div className="flex items-stretch pb-[env(safe-area-inset-bottom)]">
            {navItems.map(({ to, key, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) =>
                  `flex flex-1 flex-col items-center justify-center gap-1 py-3 text-[10px] font-bold uppercase tracking-wide transition-colors ${
                    isActive ? 'text-brand-600' : 'text-ink-300'
                  }`
                }
              >
                {({ isActive }) => (
                  <>
                    <span className={`transition-transform ${isActive ? 'scale-110' : ''}`}>
                      <Icon active={isActive} />
                    </span>
                    <span>{t(key)}</span>
                  </>
                )}
              </NavLink>
            ))}
          </div>
        </nav>
      )}
    </div>
  )
}

// ---- Inline icons -------------------------------------------------------- //
function HomeIcon({ active }: { active?: boolean }) {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
      fill={active ? 'currentColor' : 'none'} stroke="currentColor">
      <path d="M3 10.5 12 3l9 7.5" />
      <path d="M5 9.5V20a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V9.5" fill={active ? 'currentColor' : 'none'} stroke={active ? 'white' : 'currentColor'} />
    </svg>
  )
}
function CalendarIcon({ active }: { active?: boolean }) {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
      fill="none" stroke="currentColor">
      <rect x="3" y="4" width="18" height="18" rx="3" fill={active ? 'currentColor' : 'none'} />
      <path d="M3 9h18M8 3v3M16 3v3" stroke={active ? 'white' : 'currentColor'} />
    </svg>
  )
}
function ChartIcon({ active }: { active?: boolean }) {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
      fill="none" stroke="currentColor">
      <rect x="4" y="11" width="4" height="9" rx="1.5" fill={active ? 'currentColor' : 'none'} />
      <rect x="10" y="6" width="4" height="14" rx="1.5" fill={active ? 'currentColor' : 'none'} />
      <rect x="16" y="13" width="4" height="7" rx="1.5" fill={active ? 'currentColor' : 'none'} />
    </svg>
  )
}
function UserIcon({ active }: { active?: boolean }) {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
      fill={active ? 'currentColor' : 'none'} stroke="currentColor">
      <circle cx="12" cy="8" r="4" />
      <path d="M4 21c0-4 3.5-6 8-6s8 2 8 6" />
    </svg>
  )
}
function BookIcon() {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v16l-7-3-7 3V5Z" />
    </svg>
  )
}
