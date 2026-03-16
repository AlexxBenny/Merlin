import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Overview from './pages/Overview'
import Chat from './pages/Chat'
import Scheduler from './pages/Scheduler'
import MemoryPage from './pages/Memory'
import Logs from './pages/Logs'
import Config from './pages/Config'
import Missions from './pages/Missions'
import WorldState from './pages/WorldState'
import Mail from './pages/Mail'

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Overview />} />
        <Route path="/chat" element={<Chat />} />
        <Route path="/scheduler" element={<Scheduler />} />
        <Route path="/memory" element={<MemoryPage />} />
        <Route path="/logs" element={<Logs />} />
        <Route path="/config" element={<Config />} />
        <Route path="/missions" element={<Missions />} />
        <Route path="/world" element={<WorldState />} />
        <Route path="/mail" element={<Mail />} />
      </Route>
    </Routes>
  )
}
