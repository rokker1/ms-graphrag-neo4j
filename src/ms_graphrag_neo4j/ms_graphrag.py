import os
from typing import Any, Dict, List, Optional, Type
from neo4j import Driver
from openai import AsyncOpenAI

from tqdm.asyncio import tqdm, tqdm_asyncio


from ms_graphrag_neo4j.cypher_queries import *
from ms_graphrag_neo4j.utils import *
from ms_graphrag_neo4j.prompts import *


class MsGraphRAG:
    """
    A class for handling RAG (Retrieval-Augmented Generation) operations
    with Microsoft Graph data stored in Neo4j.
    """

    def __init__(
        self,
        driver: Driver,
        model: str = "gpt-4o",
        database: str = "neo4j",
    ) -> None:
        """
        Initialize MsGraphRAG with Neo4j driver and LLM.

        Args:
            driver (Driver): Neo4j driver instance
            llm (Any): Language model for generation
            database (str, optional): Neo4j database name. Defaults to "neo4j".
        """
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(
                "You need to define the `OPENAI_API_KEY` environment variable"
            )

        self._driver = driver
        self.model = model
        self._database = database
        self._openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        # Test for APOC
        try:
            self.query("CALL apoc.help('test')")
        except:
            raise ValueError("You need to install and allow APOC functions")
        # Test for GDS
        try:
            self.query("CALL gds.list('test')")
        except:
            raise ValueError("You need to install and allow GDS functions")

    async def extract_nodes_and_rels(
        self, input_texts: list, allowed_entities: list
    ) -> str:
        async def process_text(input_text):
            prompt = GRAPH_EXTRACTION_PROMPT.format(
                entity_types=allowed_entities,
                input_text=input_text,
                tuple_delimiter=";",
                record_delimiter="|",
                completion_delimiter="\n\n",
            )
            messages = [
                {"role": "user", "content": prompt},
            ]
            # Make the LLM call
            output = await self.achat(messages, model=self.model)
            # Construct JSON from output
            return parse_extraction_output(output.content)

        # Create tasks for all input texts
        tasks = [process_text(text) for text in input_texts]

        # Process tasks with tqdm progress bar
        result = []
        for task in tqdm.as_completed(
            tasks, total=len(tasks), desc="Extracting nodes & relationships"
        ):
            result.append(await task)

        total_relationships = 0
        # Import nodes and relationships
        for text, output in zip(input_texts, result):
            nodes, relationships = output
            total_relationships += len(relationships)
            # Import nodes
            self.query(
                import_nodes_query,
                params={"text": text, "chunk_id": get_hash(text), "data": nodes},
            )
            # Import relationships
            self.query(import_relationships_query, params={"data": relationships})

        return f"Successfuly extracted and imported {total_relationships} relationships"

    async def summarize_nodes_and_rels(self) -> str:
        # Summarize nodes
        nodes = self.query(candidate_nodes_summarization)

        async def process_node(node):
            messages = [
                {
                    "role": "user",
                    "content": SUMMARIZE_PROMPT.format(
                        entity_name=node["entity_name"],
                        description_list=node["description_list"],
                    ),
                },
            ]
            summary = await self.achat(messages, model=self.model)
            return {"entity": node["entity_name"], "summary": summary.content}

        # Create a progress bar for node processing
        summaries = await tqdm_asyncio.gather(
            *[process_node(node) for node in nodes], desc="Summarizing nodes"
        )

        # Summarize relationships
        rels = self.query(candidate_rels_summarization)

        async def process_rel(rel):
            entity_name = f"{rel['source']} relationship to {rel['target']}"
            messages = [
                {
                    "role": "user",
                    "content": SUMMARIZE_PROMPT.format(
                        entity_name=entity_name,
                        description_list=rel["description_list"],
                    ),
                },
            ]
            summary = await self.achat(messages, model=self.model)
            return {
                "source": rel["source"],
                "target": rel["target"],
                "summary": summary.content,
            }

        # Create a progress bar for relationship processing
        rel_summaries = await tqdm_asyncio.gather(
            *[process_rel(rel) for rel in rels], desc="Summarizing relationships"
        )

        # Import nodes
        self.query(import_entity_summary, params={"data": summaries})
        self.query(import_entity_summary_single)

        # Import relationships
        self.query(import_rel_summary, params={"data": rel_summaries})
        self.query(import_rel_summary_single)

        return "Successfuly summarized nodes and relationships"

    async def summarize_communities(self, summarize_all_levels: bool = False) -> str:
        # Calculate communities
        self.query(drop_gds_graph_query)
        self.query(create_gds_graph_query)
        community_summary = self.query(leiden_query)
        community_levels = community_summary[0]["ranLevels"]
        print(
            f"Leiden algorithm identified {community_levels} community levels "
            f"with {community_summary[0]['communityCount']} communities on the last level."
        )
        self.query(community_hierarchy_query)

        # Community summarization
        if summarize_all_levels:
            levels = list(range(community_levels))
        else:
            levels = [community_levels - 1]
        communities = self.query(community_info_query, params={"levels": levels})

        # Define async function for processing a single community
        async def process_community(community):
            input_text = f"""Entities:
                    {community['nodes']}

                    Relationships:
                    {community['rels']}"""

            messages = [
                {
                    "role": "user",
                    "content": COMMUNITY_REPORT_PROMPT.format(input_text=input_text),
                },
            ]
            summary = await self.achat(messages, model=self.model)
            return {
                "community": extract_json(summary.content),
                "communityId": community["communityId"],
            }

        # Process all communities concurrently with tqdm progress bar
        community_summary = await tqdm_asyncio.gather(
            *(process_community(community) for community in communities),
            desc="Summarizing communities",
            total=len(communities),
        )

        self.query(import_community_summary, params={"data": community_summary})
        return f"Generated {len(community_summary)} community summaries"

    def _check_driver_state(self) -> None:
        """
        Check if the Neo4j driver is still available.

        Raises:
            RuntimeError: If the Neo4j driver has been closed.
        """
        if not hasattr(self, "_driver"):
            raise RuntimeError(
                "This MsGraphRAG instance has been closed, and cannot be used anymore."
            )

    def query(
        self,
        query: str,
        params: dict = {},
        session_params: dict = {},
    ) -> List[Dict[str, Any]]:
        """Query Neo4j database.

        Args:
            query (str): The Cypher query to execute.
            params (dict): The parameters to pass to the query.
            session_params (dict): Parameters to pass to the session used for executing
                the query.

        Returns:
            List[Dict[str, Any]]: The list of dictionaries containing the query results.

        Raises:
            RuntimeError: If the connection has been closed.
        """
        self._check_driver_state()
        from neo4j import Query
        from neo4j.exceptions import Neo4jError

        if not session_params:
            try:
                data, _, _ = self._driver.execute_query(
                    Query(text=query),
                    database_=self._database,
                    parameters_=params,
                )
                return [r.data() for r in data]
            except Neo4jError as e:
                if not (
                    (
                        (  # isCallInTransactionError
                            e.code == "Neo.DatabaseError.Statement.ExecutionFailed"
                            or e.code
                            == "Neo.DatabaseError.Transaction.TransactionStartFailed"
                        )
                        and e.message is not None
                        and "in an implicit transaction" in e.message
                    )
                    or (  # isPeriodicCommitError
                        e.code == "Neo.ClientError.Statement.SemanticError"
                        and e.message is not None
                        and (
                            "in an open transaction is not possible" in e.message
                            or "tried to execute in an explicit transaction"
                            in e.message
                        )
                    )
                ):
                    raise
        # fallback to allow implicit transactions
        session_params.setdefault("database", self._database)
        with self._driver.session(**session_params) as session:
            result = session.run(Query(text=query, timeout=self.timeout), params)
            return [r.data() for r in result]

    async def achat(self, messages, model="gpt-4o", temperature=0, config={}):
        response = await self._openai_client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
            **config,
        )
        return response.choices[0].message

    def close(self) -> None:
        """
        Explicitly close the Neo4j driver connection.

        Delegates connection management to the Neo4j driver.
        """
        if hasattr(self, "_driver"):
            self._driver.close()
            # Remove the driver attribute to indicate closure
            delattr(self, "_driver")

    def __enter__(self) -> "MsGraphRAG":
        """
        Enter the runtime context for the Neo4j graph connection.

        Enables use of the graph connection with the 'with' statement.
        This method allows for automatic resource management and ensures
        that the connection is properly handled.

        Returns:
            MsGraphRAG: The current graph connection instance

        Example:
            with MsGraphRAG(...) as graph:
                graph.query(...)  # Connection automatically managed
        """
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """
        Exit the runtime context for the Neo4j graph connection.

        This method is automatically called when exiting a 'with' statement.
        It ensures that the database connection is closed, regardless of
        whether an exception occurred during the context's execution.

        Args:
            exc_type: The type of exception that caused the context to exit
                      (None if no exception occurred)
            exc_val: The exception instance that caused the context to exit
                     (None if no exception occurred)
            exc_tb: The traceback for the exception (None if no exception occurred)

        Note:
            Any exception is re-raised after the connection is closed.
        """
        self.close()

    def __del__(self) -> None:
        """
        Destructor for the Neo4j graph connection.

        This method is called during garbage collection to ensure that
        database resources are released if not explicitly closed.

        Caution:
            - Do not rely on this method for deterministic resource cleanup
            - Always prefer explicit .close() or context manager

        Best practices:
            1. Use context manager:
               with MsGraphRAG(...) as graph:
                   ...
            2. Explicitly close:
               graph = MsGraphRAG(...)
               try:
                   ...
               finally:
                   graph.close()
        """
        try:
            self.close()
        except Exception:
            # Suppress any exceptions during garbage collection
            pass
