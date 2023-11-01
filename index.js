// 1. IMPORTING DOCUMENT LOADERS
// import { CSVLoader } from "langchain/document_loaders/fs/csv";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
// import { JSONLoader } from "langchain/document_loaders/fs/json";
// import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { TextLoader } from "langchain/document_loaders/fs/text";

// 2. IMPORT OPENAI STUFF
import { RetrievalQAChain } from "langchain/chains";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { OpenAI } from "langchain/llms/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
//LOCAL VECTOR DB
import { HNSWLib } from "langchain/vectorstores/hnswlib";

// 3. IMPORT TOKEN COUNTING STUFF
import { Tiktoken } from "@dqbd/tiktoken/lite";
import { load } from "@dqbd/tiktoken/load";
import models from "@dqbd/tiktoken/model_to_encoding.json" assert { type: "json" };
import registry from "@dqbd/tiktoken/registry.json" assert { type: "json" };

// 4. ENV VAR AND FS FOR FILE STUFF
import dotenv from "dotenv";
import fs from "fs";
dotenv.config();

// 5. INIT DOCUMENT LOADER FROM FOLDER
const loader = new DirectoryLoader("./documents", {
	// ".json": (path) => new JSONLoader(path),
	".txt": (path) => new TextLoader(path),
	// ".csv": (path) => new CSVLoader(path),
	// ".pdf": (path) => new PDFLoader(path),
});

// 6. LOAD THE DOCUMENTS
console.log("Loading docs...");
const docs = await loader.load();
console.log("Docs loaded.");

// 7. CALCULATING THE PRICE OF THE EMBEDDINGS
async function calculateCost() {
	const modelName = "text-embedding-ada-002";
	const modelKey = models[modelName];
	const model = await load(registry[modelKey]);
	const encoder = new Tiktoken(
		model.bpe_ranks,
		model.special_tokens,
		model.pat_str
	);
	const tokens = encoder.encode(JSON.stringify(docs));
	const tokenCount = tokens.length;
	const ratePerThousandTokens = 0.0004;
	const cost = (tokenCount / 1000) * ratePerThousandTokens;
	encoder.free();
	return cost;
}

const VECTOR_STORE_PATH = "Documents.index";
const question = "Who is the main character?";

// 8. NORMALIZING THE DOCUMENTS
function normalizeDocuments(docs) {
	return docs.map((doc) => {
		if (typeof doc.pageContent === "string") {
			return doc.pageContent;
		} else if (Array.isArray(doc.pageContent)) {
			return doc.pageContent.join("\n");
		}
	});
}

// 9. MAIN FUNCTION TO RUN THE PRICESS
export const run = async () => {
	// 10. CALCULATE COST OF
	console.log("Calculating cost...");
	const cost = await calculateCost();
	console.log("Cost calculated:", cost);

	// 11. CHECKING IF COST IS BELOW $1
	if (cost <= 1) {
		// 12. INIT OPENAI
		const model = new OpenAI({});

		let vectorStore;

		// 13. CHECKING IF VECTOR STORE EXISTS
		console.log("Checking for existing vector store...");
		if (fs.existsSync(VECTOR_STORE_PATH)) {
			// 14. LOAD EXISTING VECTOR STORE
			console.log("Loading existing vector store...");
			//LOADING FROM DB
			vectorStore = await HNSWLib.load(
				VECTOR_STORE_PATH,
				new OpenAIEmbeddings()
			);
			console.log("Vector store loaded.");
		} else {
			// 15. CREATE NEW VECTOR STORE IF DOESNT EXIST
			console.log("Creating new vector store...");
			//SPLITTING INTO CHUNKS
			const textSplitter = new RecursiveCharacterTextSplitter({
				chunkSize: 1000,
			});
			//NORMALIZING DOC
			const normalizedDocs = normalizeDocuments(docs);
			//SPLIT DOC TO PAGES
			const splitDocs = await textSplitter.createDocuments(normalizedDocs);

			// 16. GENERATE EMBEDDINGS WITH OPENAI
			vectorStore = await HNSWLib.fromDocuments(
				splitDocs,
				new OpenAIEmbeddings()
			);
			// 17. SAVE VECTOR STORE TO INDEX
			await vectorStore.save(VECTOR_STORE_PATH);

			console.log("Vector store created.");
		}

		//CREATE A RETRIEVAL CHAIN USING THE LANGUAGE MODEL AND VECTOR STORE
		console.log("Creating retrieval chain...");
		const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

		// 19. QUERY THE RETREIVAL CHAIN WITH OUR QUESTION
		console.log("Querying chain...");
		const res = await chain.call({ query: question });
		console.log({ res });
	} else {
		// 20. IF COSTS ARE MORE THAN OUR BUDGET CONSOLE LOG
		console.log("The cost of embedding exceeds $1. Skipping embeddings.");
	}
};

// 21. Run the main function
run();
