<template>
  <div class="am-ai-chatbot">
    <TopBar title="Ask AMi" button="Go Back" url="https://australian.museum/" />
    <div class="am-ai-col-control">
      <div class="am-ai-chat-window" :class="{ 'am-ai-chat-empty': !hasAnswer }">
        <div
            class="am-ai-chat-window-control"
            ref="chatContainer"
        >
          <!-- <div v-show="chatHistory.length > 0 || sessionStorageChat.value.length > 0" v-for="(item, index) in chatHistory || sessionStorageChat.value" :key="index" class="am-ai-chat-history">
            <ChatQuestion v-if="item.role === 'user'" :hasQuestion="true" :question="item.message" />
            <ChatAnswer v-if="item.role === 'assistant'" :answer="item.message" />
            <ButtonGroup v-if="item.role === 'assistant'" :text="item.message" />
          </div> -->

          <div v-show="chatHistory.length > 0 || sessionStorageChat.length > 0" 
            v-for="(item, index) in (chatHistory.length > 0 ? chatHistory : sessionStorageChat)" 
            :key="index" 
            class="am-ai-chat-history">
            check more
          <ChatQuestion v-if="item.role === 'user'" :hasQuestion="true" :question="item.message" />
          <ChatAnswer v-if="item.role === 'assistant'" :answer="item.message" />
          <ButtonGroup v-if="item.role === 'assistant'" :text="item.message" />
        </div>
          <div ref="currentChatContainer">
            <ChatQuestion :hasQuestion="chatState.hasQuestion" :question="chatState.askedQuestion" />
            <div
                v-show="!chatState.hasAnswer && chatState.hasQuestion"
                class="am-ai-dot-flash-display"
            >
              <DotFlashing />
            </div>
            <div v-show="chatState.hasAnswer">
              <ChatAnswer :answer="chatState.streamTextDisplay" />
              <ButtonGroup :text="chatState.streamTextDisplay" />
            </div>
          </div>
        </div>
        <form @submit.prevent="submitQuestion" class="am-ai-search-bar">
          <div class="am-ai-textarea-wrapper">
            <textarea
                name="query"
                ref="questionTextareaRef"
                v-model="inputQuestion"
                @input="autoResize"
                @keydown.enter.exact.prevent="handleQuestionSubmit"
                :placeholder="chatState.hasAnswer ? 'Ask another question' : 'Ask a question'"
                aria-label="Ask your question"
                required
                rows="1"
            ></textarea>
            <button type="submit" aria-label="submit">
              <svg aria-hidden="true" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20"
                   fill="none">
                <path
                    d="M9.05198 6.7948L5.16303 10.684C4.97303 10.8738 4.75032 10.9661 4.4949 10.9608C4.23948 10.9556 4.01685 10.858 3.82698 10.6681C3.65087 10.4781 3.56018 10.2554 3.5549 10C3.54962 9.74459 3.64192 9.52189 3.83178 9.33189L9.33178 3.83189C9.42511 3.73869 9.52872 3.67057 9.64261 3.62751C9.75636 3.58445 9.87546 3.56293 9.9999 3.56293C10.1243 3.56293 10.2434 3.58445 10.3572 3.62751C10.4711 3.67057 10.5747 3.73869 10.668 3.83189L16.1728 9.33689C16.3523 9.51633 16.442 9.73557 16.442 9.99459C16.442 10.2536 16.3523 10.4781 16.1728 10.6681C15.983 10.858 15.7577 10.9529 15.497 10.9529C15.2363 10.9529 15.0109 10.858 14.8209 10.6681L10.9478 6.7948V15.8881C10.9478 16.1541 10.8564 16.3786 10.6734 16.5617C10.4904 16.7446 10.2659 16.8361 9.9999 16.8361C9.73393 16.8361 9.50942 16.7446 9.32636 16.5617C9.14344 16.3786 9.05198 16.1541 9.05198 15.8881V6.7948Z"
                    fill="#606060" />
              </svg>
            </button>
          </div>
          <p class="am-ai-info">AMi is artificial intelligence and can make mistakes. Please verify important
            information.</p>
        </form>
      </div>
      <div class="am-ai-sidebar">
        <Sources :sources="chatState.sources" />
        <FollowUpQuestions
            :questions="chatState.followupQuestions"
            @question-clicked="submitFollowUpQuestion"
        />
      </div>
    </div>
  </div>
</template>

<script setup>

import {ref, reactive, computed, onMounted, watch, nextTick, inject} from 'vue';
import TopBar from './components/TopBar.vue'
import FollowUpQuestions from './components/FollowUpQuestions.vue'
import Sources from './components/Sources.vue'
import ButtonGroup from './components/ButtonGroup.vue'
import DotFlashing from './components/DotFlashing.vue'
import ChatQuestion from './components/ChatQuestion.vue'
import ChatAnswer from './components/ChatAnswer.vue'

const questionTextareaRef = ref(null);
const inputQuestion = ref(null);
const chatHistory = ref([]);
const chatContainer = ref(null);
const currentChatContainer = ref(null);
const memoryId = ref('');
const sessionStorageChat = ref(JSON.parse(sessionStorage.getItem('chatHistoryData') || '[]'))

// Define the custom emit event for opensearch
const emitChatQuestion = (question) => {
  const event = new CustomEvent('chat-message-submit', {
    bubbles: true,
    detail: { // Custom data must be in 'detail' property
      question: question
    }
  });
  window.dispatchEvent(event);
};

const addToHistory = (question, answer) => {
  chatHistory.value.push(
    { role: "user", message: question }, { role: "assistant", message: answer }
  )
  sessionStorage.setItem('chatHistoryData', JSON.stringify(chatHistory.value))
  console.log('session', sessionStorage.getItem('chatHistoryData'));
  sessionStorageChat.value = JSON.parse(sessionStorage.getItem('chatHistoryData') || '[]');
  console.log('session', sessionStorageChat.value);
}

// State management
const chatState = reactive({
  hasQuestion: false,
  hasAnswer: false,
  askedQuestion: "",
  streamTextDisplay: "",
  followupQuestions: [],
  sources: []
});

// Scroll to message when chat history changes
watch([chatHistory, chatState], () => {
  nextTick(() => {
    if (chatContainer.value) {
      chatContainer.value.scrollTop = chatContainer.value.scrollHeight - currentChatContainer.value.scrollHeight - 10;
    }
  })
}, {deep: true})


// const fetchAnswer = async (question) => {
//   addToHistory(chatState.askedQuestion, chatState.streamTextDisplay);
//   try {
//     // Prepare form data
//     const formData = new FormData();
//     formData.append('message', question);

//     // Only add memory_id if it exists
//     if (memoryId.value) {
//       formData.append('memory_id', memoryId.value);
//     }

//     const response = await fetch('/chat/', {
//       method: 'POST',
//       body: formData
//     });

//     if (!response.ok) throw new Error('Network response was not ok');

//     const data = await response.json();
//     console.log('data', data);

//     // Extract data using more robust approach
//     const outputs = data.inference_results?.[0]?.output || [];

//     // Find memory_id
//     const memoryOutput = outputs.find(output => output.name === 'memory_id');
//     if (memoryOutput) {
//       memoryId.value = memoryOutput.result;
//     }

//     // Find the ML model output and extract text
//     const mlOutput = outputs.find(output => output.name === 'MLModelTool');
//     let text = '';

//     if (mlOutput?.result) {
//       try {
//         const parsedResult = JSON.parse(mlOutput.result);
//         text = parsedResult.output?.message?.content?.[0]?.text || '';
//       } catch (parseError) {
//         console.error('Error parsing ML output:', parseError);
//         throw new Error('Failed to parse response data');
//       }
//     } else {
//       throw new Error('No ML output found in response');
//     }
//     return convertToObject(text);
//   } catch (error) {
//     console.error('Error fetching answer:', error);
//     throw error;
//   }
// };

const fetchAnswer = async (question) => {
  try {
    let response;
    if (chatState.followupQuestions.length == 0) {
      response = await fetch('/data.json');
    } else {
      response =  await fetch('/data1.json');
    }
    await new Promise(resolve => setTimeout(resolve, 2000));
    const data = await response.json();
    console.log('data', data);
    const text = data.inference_results[0].output[2].result.output.message.content[0].text;
    return convertToObject(text);
  } catch (error) {
    console.error('Error fetching answer:', error);
    throw error;
  }
};

const convertToObject = (inputString) => {
  const result = {
    text: '',
    sources: [],
    follow_up_questions: []
  }
  if (!inputString.includes('[STARTJSON]') || !inputString.includes('[ENDJSON]')) {
    result.text = inputString.trim()
    return result;
  }
  const text = inputString.split('[STARTJSON]')[0].trim();
  const jsonStart = inputString.indexOf('[STARTJSON]') + '[STARTJSON]'.length;
  const jsonEnd = inputString.indexOf('[ENDJSON]');
  const jsonString = inputString.substring(jsonStart, jsonEnd).trim();

  let jsonData;
  try {
    jsonData = JSON.parse(jsonString);
  } catch (e) {
    console.error("Failed to parse JSON:", e);
  }
  result.text = text,
      result.sources = jsonData.sources || [],
      result.follow_up_questions = jsonData.follow_up_questions || []
  return result;
}

// Business logic
const askQuestion = async (question) => {
  if (!question?.trim()) return;
  if (chatState.askedQuestion && chatState.streamTextDisplay)
  addToHistory(chatState.askedQuestion, chatState.streamTextDisplay);
  // Reset state for new question
  Object.assign(chatState, {
    hasQuestion: true,
    hasAnswer: false,
    askedQuestion: question,
  });
  try {
    const data = await fetchAnswer(question);
    // test markdown
    receiveAnswer(data);
  } catch {
    // Handle error state if needed
    chatState.hasAnswer = false;
  }
};


const receiveAnswer = async (data) => {
  chatState.hasAnswer = true;
  chatState.streamTextDisplay = data.text;
  chatState.sources = data.sources;
  chatState.followupQuestions = data.follow_up_questions;
};

const autoResize = () => {
  if (!questionTextareaRef.value) return;
  questionTextareaRef.value.style.height = 'auto';
  questionTextareaRef.value.style.height = `${questionTextareaRef.value.scrollHeight + 2}px`;
};

// Event Enter handlers
const handleQuestionSubmit = () => {
  if (inputQuestion.value.trim()) {
    emitChatQuestion(inputQuestion.value.trim());
    askQuestion(inputQuestion.value);
    inputQuestion.value = '';
    nextTick(() => {
      autoResize();
    });
  }
};
// Normal Clicks
const submitQuestion = () => {
  if (inputQuestion.value.trim()) {
    emitChatQuestion(inputQuestion.value.trim());
    askQuestion(inputQuestion.value);
    inputQuestion.value = '';
    nextTick(() => {
      autoResize();
    });
  }
};

const submitFollowUpQuestion = (question) => {
  if (question) {
    emitChatQuestion(question);
    askQuestion(question);
  }
};

// answer url query
onMounted(() => {
  const urlParams = new URLSearchParams(window.location.search);
  const query = urlParams.get('query');
  if (query) {
    askQuestion(query);
  }
  questionTextareaRef.value.focus();
  autoResize();
  sessionStorageChat.value = JSON.parse(sessionStorage.getItem('chatHistoryData') || '[]');
  console.log('check session', sessionStorageChat.value)
  console.log(chatHistory.length > 0 || sessionStorageChat.value.length > 0)
});
</script>

<style scoped>
/* Chatbot Styles */
.am-ai-chatbot {
  margin: 12px 0;
}

.am-ai-col-control {
  display: flex;
  flex-direction: column;
  width: 100%;
  gap: 0;
  margin-top: 12px;
  margin-bottom: 12px;
}

.am-ai-chat-window {
  width: 100%;
  display: flex;
  flex-direction: column;
  height: calc(100vh - 150px);
  max-height: 580px;
  margin-bottom: 16px;
}

@media (min-width: 650px) and (max-width: 979px) {
  .am-ai-chat-window {
    max-height: 600px;
    height: calc(100vh - 120px);
  }
}

@media (max-width: 650px) {
  .am-ai-chat-window {
    max-height: 720px;
    height: calc(100vh - 100px);
  }
}

.am-ai-chat-empty {
  height: auto;
}

.am-ai-chat-window-control {
  width: 100%;
  display: flex;
  flex-direction: column;
  padding-bottom: 8px;
  padding-top: 8px;
  padding-left: 4px;
  position: relative;
  overflow-y: auto;
  padding-right: 16px;
  margin-bottom: 12px;
}

.am-ai-sidebar {
  display: flex;
  flex-direction: column;
  width: 100%;
  margin: 20px auto;
}

.am-ai-search-bar {
  position: relative;
  max-width: 100%;
  width: calc(100% - 12px);
}

.am-ai-textarea-wrapper {
  position: relative;
  margin-top: 0;
  transition: margin-top 0.2s ease;
  display: flex;
  align-items: flex-end;
}

.am-ai-textarea-wrapper button {
  background-color: transparent !important;
}

.am-ai-search-bar textarea {
  border: 1px solid rgba(0, 0, 0, 0.2);
  background: #FFF;
  width: 100%;
  padding: 10px 36px 10px 16px;
  font-size: 16px;
  box-sizing: border-box;
  border-radius: 0;
  height: 42px;
  margin-bottom: 8px;
  font-style: normal;
  overflow-y: auto;
  word-break: break-word;
  white-space: pre-wrap;
  resize: none;
  max-height: 200px;
}

.am-ai-search-bar textarea::placeholder {
  font-style: normal;
  color: #606060;
}

.am-ai-search-bar button {
  position: absolute;
  bottom: 13px;
  right: 10px;
  background-color: white;
  border: none;
  padding: 0 4px;
  margin: 0;
}

.am-ai-search-bar button svg {
  margin: 0;
}

.am-ai-info {
  font-size: 13px;
  padding: 0;
  margin: 0;
}

.am-ai-dot-flash-display {
  display: flex;
  justify-content: center;
  padding: 0;
  margin: 16px 0;
}

.am-ai-chat-history {
  margin-bottom: 16px;
}

@media (min-width: 980px) {
  .am-ai-sidebar {
    width: calc(45% - 24px);
    margin: 10px auto;
  }

  .am-ai-col-control {
    flex-direction: row;
    gap: 48px;
  }

  .am-ai-chat-window {
    width: calc(55% - 24px);
  }
}
</style>
