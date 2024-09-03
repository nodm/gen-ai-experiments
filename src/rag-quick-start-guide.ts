// https://medium.com/@eric_vaillancourt/mastering-langchain-rag-a-comprehensive-tutorial-series-part-1-28faf6257fea
// https://js.langchain.com/v0.2/docs/how_to/qa_chat_history

import {RecursiveCharacterTextSplitter} from 'langchain/text_splitter';
import {MemoryVectorStore} from 'langchain/vectorstores/memory';
import {formatDocumentsAsString} from 'langchain/util/document';
import {CheerioWebBaseLoader} from '@langchain/community/document_loaders/web/cheerio';
import {Document} from '@langchain/core/documents';
import {AIMessage, HumanMessage} from '@langchain/core/messages';
import {StringOutputParser} from '@langchain/core/output_parsers';
import {ChatPromptTemplate, MessagesPlaceholder} from '@langchain/core/prompts';
import {Runnable, RunnableConfig, RunnablePassthrough, RunnableSequence} from '@langchain/core/runnables';
import {Ollama, OllamaEmbeddings} from '@langchain/ollama';
import {OpenAIEmbeddings, ChatOpenAI, OpenAI} from '@langchain/openai';
import config from './config.js';

type ChainParameters = {
  chat_history?: (HumanMessage | AIMessage)[];
  context?: string;
  question: string;
};

const chat_history: (HumanMessage | AIMessage)[] = [];

export async function ragQuickStartGuide({llm, embeddings} = getLLM('ollama')) {
  const splittedDocuments = await loadDocuments('https://lilianweng.github.io/posts/2023-06-23-agent/');

  const retriever = await getVectorStoreRetriever(splittedDocuments, embeddings);

  const contextualizeQPrompt = ChatPromptTemplate.fromMessages([
    [
      'system',
      `
        Given a chat history and the latest user question which might reference context in the chat history,
        formulate a standalone question which can be understood without the chat history.
        Do not answer the question, just reformulate it if needed and otherwise return it as is.
        Provide just a question, no explanation and other information.
      `,
    ],
    new MessagesPlaceholder('chat_history'),
    ['human', '{question}'],
  ]);
  const contextualizeQChain = contextualizeQPrompt.pipe(llm as Ollama).pipe(new StringOutputParser());
  const contextualizedQuestion = (input: ChainParameters) => {
    if ('chat_history' in input) {
      return contextualizeQChain;
    }

    return input.question as unknown as Runnable<string, string, RunnableConfig>;
  };

  const qaPrompt = ChatPromptTemplate.fromMessages([
    [
      'system',
      `
        You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise.

        {context}
      `,
    ],
    new MessagesPlaceholder('chat_history'),
    ['human', '{question}'],
  ]);

  const ragChain = RunnableSequence.from([
    RunnablePassthrough.assign({
      context: async (input: ChainParameters) => {
        if (!('chat_history' in input)) return '';

        const chain = contextualizedQuestion(input);

        return chain.pipe(retriever).pipe(formatDocumentsAsString);
      },
    }),
    qaPrompt,
    llm,
    new StringOutputParser(),
  ]);

  await askQuestion(ragChain, 'What is task decomposition?');
  await askQuestion(ragChain, 'What are common ways of doing it?');
  await askQuestion(ragChain, 'What are the complemented components of a LLM-powered autonomous agent system?');
  await askQuestion(ragChain, 'What is the role of the agent in the agent-environment interaction?');
  await askQuestion(ragChain, 'What is the Self-reflection?');
}

function getLLM(provider: 'ollama' | 'OpenAI') {
  switch (provider) {
    case 'ollama':
      return {
        llm: new Ollama({
          baseUrl: config.baseUrl,
          model: config.modelName,
          temperature: config.temperature,
          maxRetries: config.maxRetries,
        }),
        // https://js.langchain.com/v0.2/docs/integrations/text_embedding/ollama/
        // https://ollama.com/blog/embedding-models
        embeddings: new OllamaEmbeddings({
          model: 'mxbai-embed-large',
        }),
      };
    case 'OpenAI':
      return {
        llm: new ChatOpenAI({
          model: 'gpt-3.5-turbo',
          temperature: 0.2,
        }),
        embeddings: new OpenAIEmbeddings(),
      };
    default:
      throw new Error(`Unknown provider: ${provider}`);
  }
}

async function loadDocuments(url: string) {
  const loader = new CheerioWebBaseLoader(url);
  const docs = await loader.load();

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1_000,
    chunkOverlap: 200,
  });
  const splittedDocuments = await textSplitter.splitDocuments(docs);

  return splittedDocuments;
}

async function getVectorStoreRetriever(
  splittedDocuments: Document<Record<string, unknown>>[],
  embeddings: OllamaEmbeddings | OpenAIEmbeddings
) {
  // https://js.langchain.com/v0.2/docs/integrations/vectorstores/memory/
  const vectorStore = await MemoryVectorStore.fromDocuments(splittedDocuments, embeddings);

  const retriever = vectorStore.asRetriever({
    searchType: 'mmr',
    searchKwargs: {
      fetchK: 10,
    },
    // filter,
    k: 4,
  });

  return retriever;
}

async function askQuestion(runnableSequence: RunnableSequence<ChainParameters, string>, question: string) {
  const stream = await runnableSequence.stream({
    chat_history,
    question,
  });

  console.log(`\x1b[32m${question}\x1b[90m`);

  let answer = '';
  for await (const chunk of stream) {
    process.stdout.write(chunk);
    answer = answer.concat(chunk);
  }
  process.stdout.write('\n\n\x1b[0m');

  chat_history.push(new HumanMessage({content: question}), new AIMessage({content: answer}));
}
