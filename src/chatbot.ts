import {InMemoryChatMessageHistory} from '@langchain/core/chat_history';
import {AIMessage, HumanMessage} from '@langchain/core/messages';
import {StringOutputParser} from '@langchain/core/output_parsers';
import {ChatPromptTemplate} from '@langchain/core/prompts';
import {RunnableWithMessageHistory} from '@langchain/core/runnables';
import {Ollama} from '@langchain/ollama';
import config from './config.js';

const messages = [
  new HumanMessage({content: "hi! I'm bob"}),
  new AIMessage({content: 'hi!'}),
  new HumanMessage({content: 'I like vanilla ice cream'}),
  new AIMessage({content: 'nice'}),
  new HumanMessage({content: 'whats 2 + 2'}),
  new AIMessage({content: '4'}),
  new HumanMessage({content: 'thanks'}),
  new AIMessage({content: 'No problem!'}),
  new HumanMessage({content: 'having fun?'}),
  new AIMessage({content: 'yes!'}),
  new HumanMessage({content: "That's great!"}),
  new AIMessage({content: 'yes it is!'}),
];

export async function chatbot(options?: {modelName?: string; temperature?: number; maxRetries?: number}) {
  const model = new Ollama({
    model: options?.modelName ?? config.modelName,
    temperature: options?.temperature ?? config.temperature,
    maxRetries: options?.maxRetries ?? config.maxRetries,
  });

  const prompt = ChatPromptTemplate.fromMessages([
    ['system', 'You are a helpful assistant who remembers all details the user shares with you.'],
    ['placeholder', '{chat_history}'],
    ['human', '{input}'],
  ]);

  const chatConfig = {
    configurable: {
      sessionId: 'Bob',
    },
  };

  const chain = prompt.pipe(model).pipe(new StringOutputParser());

  const messageHistories = new Map<string, InMemoryChatMessageHistory>();
  const withMessageHistory = new RunnableWithMessageHistory({
    runnable: chain,
    getMessageHistory: async sessionId => {
      if (!messageHistories.has(sessionId)) {
        const messageHistory = new InMemoryChatMessageHistory();
        await messageHistory.addMessages(messages.slice(-10)); // Limit the number of messages to 10
        messageHistories.set(sessionId, messageHistory);
      }

      return messageHistories.get(sessionId)!;
    },
    inputMessagesKey: 'input',
    historyMessagesKey: 'chat_history',
  });

  const stream = await withMessageHistory.stream(
    {
      input: 'Hi! What is my name?',
    },
    chatConfig
  );

  for await (const chunk of stream) {
    process.stdout.write(chunk);
  }
  process.stdout.write('\n');
}
