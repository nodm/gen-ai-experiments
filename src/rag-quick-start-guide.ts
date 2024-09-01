// https://medium.com/@eric_vaillancourt/mastering-langchain-rag-a-comprehensive-tutorial-series-part-1-28faf6257fea
// https://js.langchain.com/v0.2/docs/how_to/qa_chat_history

import {createStuffDocumentsChain} from 'langchain/chains/combine_documents';
import {pull} from 'langchain/hub';
import {RecursiveCharacterTextSplitter} from 'langchain/text_splitter';
import {MemoryVectorStore} from 'langchain/vectorstores/memory';
import {formatDocumentsAsString} from 'langchain/util/document';
import {CheerioWebBaseLoader} from '@langchain/community/document_loaders/web/cheerio';
import {AIMessage, HumanMessage, SystemMessage} from '@langchain/core/messages';
import {StringOutputParser} from '@langchain/core/output_parsers';
import {ChatPromptTemplate, MessagesPlaceholder} from '@langchain/core/prompts';
import {Runnable, RunnableConfig, RunnablePassthrough, RunnableSequence} from '@langchain/core/runnables';
import {Ollama, OllamaEmbeddings} from '@langchain/ollama';
import config from './config.js';

export async function ragQuickStartGuide() {
  const llm = new Ollama({
    model: config.modelName,
    temperature: config.temperature,
    maxRetries: config.maxRetries,
  });

  // https://js.langchain.com/v0.2/docs/integrations/text_embedding/ollama/
  // https://ollama.com/blog/embedding-models
  const embeddings = new OllamaEmbeddings({
    model: 'mxbai-embed-large',
  });

  const loader = new CheerioWebBaseLoader('https://lilianweng.github.io/posts/2023-06-23-agent/');
  const docs = await loader.load();

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const splits = await textSplitter.splitDocuments(docs);

  // https://js.langchain.com/v0.2/docs/integrations/vectorstores/memory/
  const vectorStore = await MemoryVectorStore.fromDocuments(splits, embeddings);

  const retriever = vectorStore.asRetriever({
    searchType: 'mmr',
    searchKwargs: {
      fetchK: 10,
    },
    // filter,
    k: 2,
  });

  const prompt = await pull<ChatPromptTemplate>('rlm/rag-prompt');

  // const ragChain = await createStuffDocumentsChain({
  //   llm,
  //   prompt,
  //   outputParser: new StringOutputParser(),
  // });

  // await ragChain.invoke({
  //   context: await retriever.invoke('What is Task Decomposition?'),
  //   question: 'What is Task Decomposition?',
  // });

  const contextualizeQSystemPrompt = `Given a chat history and the latest user question
which might reference context in the chat history, formulate a standalone question
which can be understood without the chat history. Do NOT answer the question,
just reformulate it if needed and otherwise return it as is.`;
  const contextualizeQPrompt = ChatPromptTemplate.fromMessages([
    ['system', contextualizeQSystemPrompt],
    new MessagesPlaceholder('chat_history'),
    ['human', '{question}'],
  ]);
  const contextualizeQChain = contextualizeQPrompt.pipe(llm).pipe(new StringOutputParser());
  // const q = await contextualizeQChain.invoke({
  //   chat_history: [new HumanMessage('What does LLM stand for?'), new AIMessage('Large language model')],
  //   question: 'What is meant by large',
  // });
  // console.log(q);

  const qaSystemPrompt = `You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

{context}`;
  const qaPrompt = ChatPromptTemplate.fromMessages([
    new SystemMessage({content: qaSystemPrompt}),
    new MessagesPlaceholder('chat_history'),
    new HumanMessage({content: '{question}'}),
  ]);

  const contextualizedQuestion = (input: Record<string, unknown>) => {
    if ('chat_history' in input) {
      return contextualizeQChain;
    }
    return input.question as Runnable<string, string, RunnableConfig>;
  };

  const ragChain = RunnableSequence.from([
    RunnablePassthrough.assign({
      context: async (input: Record<string, unknown>) => {
        if ('chat_history' in input) {
          const chain = contextualizedQuestion(input);
          return chain.pipe(retriever).pipe(formatDocumentsAsString);
        }
        return '';
      },
    }),
    qaPrompt,
    llm,
    new StringOutputParser(),
  ]);

  const chat_history: string[] = [];

  const question = 'What is task decomposition?';
  const aiMsg = await ragChain.invoke({question, chat_history});

  console.log(aiMsg);

  chat_history.push(aiMsg);

  const secondQuestion = 'What are common ways of doing it?';
  // await ragChain.invoke({question: secondQuestion, chat_history});

  const stream = await ragChain.stream({
    question: secondQuestion,
    chat_history,
  });

  for await (const chunk of stream) {
    process.stdout.write(chunk);
  }
  process.stdout.write('\n');
}
