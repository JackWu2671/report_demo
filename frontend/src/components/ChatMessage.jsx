import React from 'react'
import WorkflowSteps from './WorkflowSteps'

const UserAvatar = () => (
  <div className="msg-avatar msg-avatar--user">
    <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 12c2.7 0 4.8-2.1 4.8-4.8S14.7 2.4 12 2.4 7.2 4.5 7.2 7.2 9.3 12 12 12zm0 2.4c-3.2 0-9.6 1.6-9.6 4.8v2.4h19.2v-2.4c0-3.2-6.4-4.8-9.6-4.8z"/>
    </svg>
  </div>
)

const AssistantAvatar = () => (
  <div className="msg-avatar msg-avatar--assistant">
    <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
      <path d="M21 3H3c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h18c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-9 9l-5 3V9l5 3z"/>
    </svg>
  </div>
)

export default function ChatMessage({ message, isStreaming }) {
  const isUser = message.role === 'user'
  const { content = '', steps = [] } = message

  const hasSteps = steps.length > 0
  const hasContent = content.trim().length > 0
  const showTyping = !isUser && isStreaming && !hasContent && !hasSteps

  return (
    <div className={`msg-row msg-row--${isUser ? 'user' : 'assistant'}`}>
      {!isUser && <AssistantAvatar />}
      <div className="msg-bubble-wrap">
        {/* 工作流步骤进度（仅 assistant） */}
        {!isUser && hasSteps && (
          <WorkflowSteps steps={steps} />
        )}

        {/* 消息内容气泡 */}
        {(hasContent || showTyping) && (
          <div className={`msg-bubble msg-bubble--${isUser ? 'user' : 'assistant'}`}>
            {showTyping
              ? <span className="msg-typing"><span /><span /><span /></span>
              : content}
          </div>
        )}
      </div>
      {isUser && <UserAvatar />}
    </div>
  )
}
