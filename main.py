#!/usr/bin/env python3
"""
Knowledge Graph Builder - Llama-Based Visual Learning Tool
Creates visual knowledge graphs from course materials
Author: Pranay M
"""

import ollama
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
from rich.markdown import Markdown
import json
from typing import List, Dict

console = Console()

GRAPH_TYPES = ["Concept Map", "Mind Map", "Hierarchical", "Network", "Timeline", "Flowchart"]


class KnowledgeGraphBuilder:
    def __init__(self, model: str = "llama3.2"):
        self.model = model
        self.graphs = []
    
    def extract_entities(self, content: str) -> dict:
        prompt = f"""Extract key entities and relationships from this educational content.

Content:
{content}

Return JSON:
{{
    "entities": [
        {{
            "id": "E1",
            "name": "entity name",
            "type": "concept/person/event/theory/term",
            "description": "brief description",
            "importance": "high/medium/low"
        }}
    ],
    "relationships": [
        {{
            "source": "E1",
            "target": "E2",
            "relationship": "type of connection",
            "description": "how they relate",
            "strength": "strong/moderate/weak"
        }}
    ],
    "main_topic": "central theme",
    "subtopics": ["related subtopics"],
    "hierarchy": {{
        "top_level": ["main concepts"],
        "second_level": {{"concept": ["sub-concepts"]}},
        "details": {{"sub-concept": ["specifics"]}}
    }}
}}"""

        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        return self._parse_json(response['message']['content'])
    
    def build_concept_map(self, topic: str, entities: dict = None) -> dict:
        prompt = f"""Create a concept map structure for: {topic}

{f"Using these entities: {json.dumps(entities, indent=2)[:1000]}" if entities else ""}

Return JSON:
{{
    "concept_map": {{
        "central_concept": "{topic}",
        "main_branches": [
            {{
                "branch": "main concept",
                "sub_concepts": [
                    {{
                        "concept": "sub-concept",
                        "connection_phrase": "linking words",
                        "details": ["specific details"]
                    }}
                ]
            }}
        ],
        "cross_links": [
            {{
                "from": "concept A",
                "to": "concept B",
                "relationship": "how connected"
            }}
        ]
    }},
    "mermaid_code": "graph TD\\n  A[Central] --> B[Branch1]\\n  A --> C[Branch2]",
    "text_representation": "ASCII art or text-based visualization",
    "study_path": ["recommended order to learn concepts"],
    "key_connections": ["most important relationships to understand"]
}}"""

        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        result = self._parse_json(response['message']['content'])
        self.graphs.append(result)
        return result
    
    def build_mind_map(self, central_idea: str) -> dict:
        prompt = f"""Create a mind map structure for: {central_idea}

Return JSON:
{{
    "mind_map": {{
        "center": "{central_idea}",
        "branches": [
            {{
                "branch_name": "main branch",
                "color_suggestion": "color",
                "icon_suggestion": "emoji",
                "sub_branches": [
                    {{
                        "name": "sub-topic",
                        "keywords": ["key terms"],
                        "images": ["suggested visuals"]
                    }}
                ]
            }}
        ]
    }},
    "mermaid_code": "mindmap\\n  root({central_idea})\\n    Branch1\\n      Sub1\\n      Sub2",
    "visual_tips": ["how to make it memorable"],
    "color_coding": {{"category": "suggested color"}},
    "memory_anchors": ["visual/verbal memory aids"]
}}"""

        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        result = self._parse_json(response['message']['content'])
        self.graphs.append(result)
        return result
    
    def build_timeline(self, topic: str, events: List[str] = None) -> dict:
        prompt = f"""Create a timeline knowledge graph for: {topic}

{f"Events to include: {events}" if events else ""}

Return JSON:
{{
    "timeline": {{
        "topic": "{topic}",
        "time_span": "start to end period",
        "events": [
            {{
                "date": "date/period",
                "event": "what happened",
                "significance": "why important",
                "connections": ["related events"],
                "category": "political/social/scientific/etc"
            }}
        ]
    }},
    "mermaid_code": "timeline\\n  title {topic}\\n  section Period1\\n    Event1 : description",
    "cause_effect_chains": [
        {{
            "cause": "event A",
            "effect": "event B",
            "mechanism": "how A led to B"
        }}
    ],
    "turning_points": ["most significant moments"],
    "patterns": ["recurring themes or patterns"]
}}"""

        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        result = self._parse_json(response['message']['content'])
        self.graphs.append(result)
        return result
    
    def build_flowchart(self, process: str) -> dict:
        prompt = f"""Create a flowchart for this process: {process}

Return JSON:
{{
    "flowchart": {{
        "process_name": "{process}",
        "start": "starting point",
        "steps": [
            {{
                "step_id": "S1",
                "action": "what happens",
                "type": "process/decision/input/output",
                "next": ["possible next steps"],
                "conditions": "if decision, what conditions"
            }}
        ],
        "end": "end point(s)"
    }},
    "mermaid_code": "flowchart TD\\n  A[Start] --> B{{Decision}}\\n  B -->|Yes| C[Process]\\n  B -->|No| D[Other]",
    "critical_steps": ["most important steps"],
    "common_errors": ["where mistakes happen"],
    "optimization_tips": ["how to do this efficiently"]
}}"""

        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        result = self._parse_json(response['message']['content'])
        self.graphs.append(result)
        return result
    
    def generate_study_guide(self, graph: dict) -> dict:
        prompt = f"""Generate a study guide based on this knowledge graph.

Graph Data:
{json.dumps(graph, indent=2)[:2000]}

Return JSON:
{{
    "study_guide": {{
        "title": "guide title",
        "learning_objectives": ["what you'll learn"],
        "prerequisite_knowledge": ["what you should know first"],
        "estimated_time": "study duration"
    }},
    "section_by_section": [
        {{
            "section": "topic",
            "key_points": ["main ideas"],
            "practice_questions": ["self-test questions"],
            "memory_aids": ["mnemonics or tricks"]
        }}
    ],
    "connections_to_remember": ["crucial relationships"],
    "common_exam_topics": ["frequently tested areas"],
    "review_checklist": ["items to verify understanding"]
}}"""

        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        return self._parse_json(response['message']['content'])
    
    def _parse_json(self, content: str) -> dict:
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
        except:
            pass
        return {"raw_response": content}


def display_menu():
    table = Table(title="🕸️ Knowledge Graph Builder", show_header=True)
    table.add_column("Option", style="cyan", width=6)
    table.add_column("Feature", style="green")
    table.add_column("Description", style="white")
    
    table.add_row("1", "Extract Entities", "Find concepts from text")
    table.add_row("2", "Concept Map", "Build concept map")
    table.add_row("3", "Mind Map", "Create mind map")
    table.add_row("4", "Timeline", "Build timeline graph")
    table.add_row("5", "Flowchart", "Create process flowchart")
    table.add_row("6", "Study Guide", "Generate from graph")
    table.add_row("7", "View Graphs", "See saved graphs")
    table.add_row("0", "Exit", "Close application")
    
    console.print(table)


def main():
    console.print(Panel.fit(
        "[bold blue]🕸️ Knowledge Graph Builder[/bold blue]\n"
        "[green]AI-Powered Visual Learning Tool[/green]\n"
        "[dim]Author: Pranay M[/dim]",
        border_style="blue"
    ))
    
    builder = KnowledgeGraphBuilder()
    
    while True:
        display_menu()
        console.print(f"[dim]Saved Graphs: {len(builder.graphs)}[/dim]")
        
        choice = Prompt.ask("\n[cyan]Select option[/cyan]", default="0")
        
        if choice == "0":
            console.print("[yellow]Goodbye! Visualize knowledge! 🕸️[/yellow]")
            break
        
        elif choice == "7":
            if builder.graphs:
                for i, g in enumerate(builder.graphs[-3:], 1):
                    topic = g.get('concept_map', g.get('mind_map', g.get('timeline', {}))).get('central_concept', 
                            g.get('concept_map', g.get('mind_map', g.get('timeline', {}))).get('center', 'Unknown'))
                    console.print(f"  {i}: {topic}")
            else:
                console.print("[dim]No graphs saved yet.[/dim]")
            continue
        
        with console.status("[bold green]Building knowledge graph..."):
            if choice == "1":
                console.print("[dim]Paste content (end with 'EOF'):[/dim]")
                lines = []
                while True:
                    line = input()
                    if line.strip() == "EOF":
                        break
                    lines.append(line)
                result = builder.extract_entities("\n".join(lines))
                console.print(Panel(Markdown(f"```json\n{json.dumps(result, indent=2)}\n```"),
                                   title="🔍 Extracted Entities"))
            
            elif choice == "2":
                topic = Prompt.ask("Topic for concept map")
                result = builder.build_concept_map(topic)
                console.print(Panel(Markdown(f"```json\n{json.dumps(result, indent=2)}\n```"),
                                   title="🗺️ Concept Map"))
                if result.get('mermaid_code'):
                    console.print("\n[bold]Mermaid Code (paste into mermaid.live):[/bold]")
                    console.print(f"```\n{result.get('mermaid_code')}\n```")
            
            elif choice == "3":
                idea = Prompt.ask("Central idea for mind map")
                result = builder.build_mind_map(idea)
                console.print(Panel(Markdown(f"```json\n{json.dumps(result, indent=2)}\n```"),
                                   title="🧠 Mind Map"))
            
            elif choice == "4":
                topic = Prompt.ask("Topic for timeline")
                result = builder.build_timeline(topic)
                console.print(Panel(Markdown(f"```json\n{json.dumps(result, indent=2)}\n```"),
                                   title="📅 Timeline"))
            
            elif choice == "5":
                process = Prompt.ask("Process to flowchart")
                result = builder.build_flowchart(process)
                console.print(Panel(Markdown(f"```json\n{json.dumps(result, indent=2)}\n```"),
                                   title="📊 Flowchart"))
            
            elif choice == "6":
                if builder.graphs:
                    result = builder.generate_study_guide(builder.graphs[-1])
                    console.print(Panel(Markdown(f"```json\n{json.dumps(result, indent=2)}\n```"),
                                       title="📚 Study Guide"))
                else:
                    console.print("[yellow]Create a graph first.[/yellow]")
        
        console.print("\n" + "="*50)


if __name__ == "__main__":
    main()
