import { Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import Home from './pages/Home'
import Session from './pages/Session'
import Recovery from './pages/Recovery'
import WeeklyPlan from './pages/WeeklyPlan'
import Mastery from './pages/Mastery'
import Profile from './pages/Profile'

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/session" element={<Session />} />
        <Route path="/recovery" element={<Recovery />} />
        <Route path="/week" element={<WeeklyPlan />} />
        <Route path="/mastery" element={<Mastery />} />
        <Route path="/profile" element={<Profile />} />
      </Routes>
    </Layout>
  )
}
