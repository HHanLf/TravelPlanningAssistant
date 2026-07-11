import { motion } from 'framer-motion'
import { Brain, Check, Database, FileCheck2, Loader2, MessageSquareText, Search, Wand2, Wrench } from 'lucide-react'

const WORKFLOW = [
  { key: 'planner', label: 'Planner', icon: Brain },
  { key: 'research', label: 'Research', icon: Search },
  { key: 'tools', label: 'Tool Calling', icon: Wrench },
  { key: 'memory', label: 'Memory', icon: Database },
  { key: 'reflection', label: 'Reflection', icon: FileCheck2 },
  { key: 'answer', label: 'Generate Answer', icon: Wand2 },
]

export function WorkflowPanel({ loading, state = {} }) {
  const hasAnswer = Boolean(state.answer || state.final_answer)
  const hasPlan = Boolean(state.plan && Object.keys(state.plan).length)
  const hasResearch = Boolean(state.knowledge_summary && Object.keys(state.knowledge_summary).length)
  const hasTools = Boolean(state.tool_results?.items?.length)
  const hasMemory = Boolean(state.memory_context && Object.keys(state.memory_context).length)
  const hasReflection = Boolean(state.reflection_result && Object.keys(state.reflection_result).length)

  function statusFor(step) {
    if (step.key === 'planner') return hasPlan ? 'done' : loading ? 'active' : 'idle'
    if (step.key === 'research') return hasResearch ? 'done' : loading && hasPlan ? 'active' : 'idle'
    if (step.key === 'tools') return hasTools ? 'done' : loading && (hasPlan || hasResearch) ? 'active' : 'idle'
    if (step.key === 'memory') return hasMemory ? 'done' : loading ? 'active' : 'idle'
    if (step.key === 'reflection') return hasReflection ? 'done' : loading && hasTools ? 'active' : 'idle'
    return hasAnswer ? 'done' : loading ? 'active' : 'idle'
  }

  return (
    <section className="workflow-card">
      <div className="section-heading">
        <div>
          <p>Agent Workflow</p>
          <h2>执行进度</h2>
        </div>
        <MessageSquareText size={18} />
      </div>

      <div className="workflow-list">
        {WORKFLOW.map((step, index) => {
          const Icon = step.icon
          const status = statusFor(step)
          return (
            <motion.div
              className={`workflow-step workflow-step--${status}`}
              key={step.key}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.04 }}
            >
              <div className="workflow-step__icon">
                {status === 'done' ? <Check size={15} /> : status === 'active' ? <Loader2 size={15} className="spin" /> : <Icon size={15} />}
              </div>
              <div>
                <strong>{step.label}</strong>
                <span>{status === 'done' ? '已完成' : status === 'active' ? '处理中' : '等待中'}</span>
              </div>
            </motion.div>
          )
        })}
      </div>
    </section>
  )
}

