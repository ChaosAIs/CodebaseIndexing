"""
Advanced Named Entity Recognition for code queries.

This module provides sophisticated entity extraction capabilities that work
with or without spaCy, focusing on code-specific entities and technical concepts.
"""

import re
import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger

try:
    import spacy
    from spacy.matcher import Matcher
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available, using pattern-based NER only")


class EntityType(Enum):
    """Enhanced entity types for code analysis."""
    # People and Organizations
    PERSON = "person"
    ORGANIZATION = "organization"
    COMPANY = "company"
    
    # Code Entities
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    INTERFACE = "interface"
    MODULE = "module"
    PACKAGE = "package"
    FILE = "file"
    VARIABLE = "variable"
    CONSTANT = "constant"
    PARAMETER = "parameter"
    
    # Technical Concepts
    TECHNOLOGY = "technology"
    FRAMEWORK = "framework"
    LIBRARY = "library"
    DATABASE = "database"
    API = "api"
    PROTOCOL = "protocol"
    PATTERN = "pattern"
    CONCEPT = "concept"
    
    # Architecture
    COMPONENT = "component"
    SERVICE = "service"
    LAYER = "layer"
    SYSTEM = "system"


@dataclass
class ExtractedEntity:
    """Represents an entity extracted from text."""
    text: str
    entity_type: EntityType
    confidence: float
    start_pos: int
    end_pos: int
    context: str = ""
    aliases: List[str] = None
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []
        if self.properties is None:
            self.properties = {}


class AdvancedNERExtractor:
    """
    Advanced Named Entity Recognition extractor for code queries.
    
    Features:
    - spaCy-based NER for general entities
    - Pattern-based extraction for code-specific entities
    - Technology and framework recognition
    - Context-aware entity disambiguation
    - Confidence scoring and ranking
    """
    
    def __init__(self):
        """Initialize the NER extractor."""
        self.nlp = None
        self.matcher = None
        self._load_models()
        self._init_patterns()
        self._init_knowledge_base()
    
    def _load_models(self):
        """Load spaCy models if available."""
        if SPACY_AVAILABLE:
            try:
                # Try different models in order of preference
                models_to_try = ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]
                
                for model_name in models_to_try:
                    try:
                        self.nlp = spacy.load(model_name)
                        self.matcher = Matcher(self.nlp.vocab)
                        logger.info(f"Loaded spaCy model: {model_name}")
                        break
                    except OSError:
                        continue
                
                if self.nlp is None:
                    logger.warning("No spaCy models found, using pattern-based extraction only")
                    
            except Exception as e:
                logger.error(f"Error loading spaCy models: {e}")
    
    def _init_patterns(self):
        """Initialize regex patterns for entity extraction."""
        self.patterns = {
            # Programming Languages
            EntityType.TECHNOLOGY: [
                r'\b(Python|JavaScript|TypeScript|Java|C\+\+|C#|Go|Rust|Ruby|PHP|Swift|Kotlin|Scala|Clojure|Haskell|Erlang|Elixir)\b',
                r'\b(HTML|CSS|SQL|GraphQL|YAML|JSON|XML|Markdown)\b',
            ],
            
            # Frameworks and Libraries
            EntityType.FRAMEWORK: [
                r'\b(React|Vue|Angular|Svelte|Next\.js|Nuxt\.js|Gatsby)\b',
                r'\b(Django|Flask|FastAPI|Express|Koa|Nest\.js|Spring|Laravel|Rails|Phoenix)\b',
                r'\b(TensorFlow|PyTorch|Keras|Scikit-learn|Pandas|NumPy|Matplotlib)\b',
                r'\b(Bootstrap|Tailwind|Material-UI|Ant Design|Chakra UI)\b',
            ],
            
            # Databases
            EntityType.DATABASE: [
                r'\b(PostgreSQL|MySQL|SQLite|MongoDB|Redis|Cassandra|DynamoDB|Neo4j|Qdrant|Elasticsearch|InfluxDB)\b',
                r'\b(Oracle|SQL Server|MariaDB|CouchDB|RethinkDB|ArangoDB)\b',
            ],
            
            # Cloud and Infrastructure
            EntityType.TECHNOLOGY: [
                r'\b(AWS|Azure|GCP|Google Cloud|DigitalOcean|Heroku|Vercel|Netlify)\b',
                r'\b(Docker|Kubernetes|Terraform|Ansible|Jenkins|GitLab CI|GitHub Actions)\b',
                r'\b(Nginx|Apache|HAProxy|Cloudflare|CDN)\b',
            ],
            
            # Code Entities - Functions
            EntityType.FUNCTION: [
                r'\b([a-z_][a-z0-9_]*)\s*\(',  # function_name(
                r'\bfunction\s+([a-z_][a-z0-9_]*)\b',  # function function_name
                r'\bdef\s+([a-z_][a-z0-9_]*)\b',  # def function_name
                r'\basync\s+def\s+([a-z_][a-z0-9_]*)\b',  # async def function_name
                r'\b([a-z_][a-z0-9_]*)\s*=>\s*',  # arrow functions
            ],
            
            # Code Entities - Classes
            EntityType.CLASS: [
                r'\bclass\s+([A-Z][a-zA-Z0-9_]*)\b',  # class ClassName
                r'\b([A-Z][a-zA-Z0-9_]*)\s+class\b',  # ClassName class
                r'\binterface\s+([A-Z][a-zA-Z0-9_]*)\b',  # interface InterfaceName
                r'\btype\s+([A-Z][a-zA-Z0-9_]*)\b',  # type TypeName
            ],
            
            # Files and Modules
            EntityType.FILE: [
                r'\b([a-z_][a-z0-9_]*\.(py|js|ts|jsx|tsx|java|cpp|c|h|go|rs|rb|php|swift|kt))\b',
                r'\b([A-Z][a-zA-Z0-9_]*\.(py|js|ts|jsx|tsx|java|cpp|c|h|go|rs|rb|php|swift|kt))\b',
            ],
            
            # Variables and Constants
            EntityType.VARIABLE: [
                r'\bvar\s+([a-z_][a-z0-9_]*)\b',
                r'\blet\s+([a-z_][a-z0-9_]*)\b',
                r'\bconst\s+([a-z_][a-z0-9_]*)\b',
                r'\b([A-Z_][A-Z0-9_]*)\b',  # CONSTANTS
            ],
            
            # API and Protocols
            EntityType.API: [
                r'\b(REST|GraphQL|gRPC|WebSocket|HTTP|HTTPS|FTP|SMTP|TCP|UDP)\b',
                r'\b(API|endpoint|route|middleware|handler)\b',
            ],
            
            # Design Patterns
            EntityType.PATTERN: [
                r'\b(Singleton|Factory|Observer|Strategy|Command|Adapter|Decorator|Facade|Proxy|MVC|MVP|MVVM)\b',
                r'\b(Repository|Service|Controller|Model|View|Component|Module|Plugin)\b',
            ],
            
            # Architecture Concepts
            EntityType.CONCEPT: [
                r'\b(microservices|monolith|serverless|event-driven|message queue|pub-sub|CQRS|DDD)\b',
                r'\b(authentication|authorization|OAuth|JWT|session|cookie|CORS|CSRF)\b',
                r'\b(caching|load balancing|scaling|performance|optimization|monitoring|logging)\b',
            ],
        }
    
    def _init_knowledge_base(self):
        """Initialize knowledge base for entity disambiguation."""
        self.knowledge_base = {
            # Common company/organization names in tech
            "companies": {
                "Apple", "Google", "Microsoft", "Amazon", "Meta", "Facebook", "Netflix", "Tesla",
                "OpenAI", "Anthropic", "Hugging Face", "GitHub", "GitLab", "Atlassian", "Slack",
                "Stripe", "Shopify", "Spotify", "Uber", "Airbnb", "Twitter", "LinkedIn", "Reddit"
            },
            
            # Common person names in tech (can be expanded)
            "tech_people": {
                "Linus Torvalds", "Tim Cook", "Elon Musk", "Jeff Bezos", "Mark Zuckerberg",
                "Satya Nadella", "Sundar Pichai", "Sam Altman", "Dario Amodei", "Yann LeCun"
            },
            
            # Technology aliases
            "tech_aliases": {
                "JS": "JavaScript",
                "TS": "TypeScript",
                "CSS3": "CSS",
                "HTML5": "HTML",
                "ES6": "JavaScript",
                "Node": "Node.js",
                "React Native": "React",
                "Vue.js": "Vue",
                "Angular.js": "Angular",
            }
        }
    
    async def extract_entities(self, text: str, context: str = "") -> List[ExtractedEntity]:
        """
        Extract entities from text using multiple approaches.
        
        Args:
            text: Input text to analyze
            context: Additional context for disambiguation
            
        Returns:
            List of extracted entities with confidence scores
        """
        entities = []
        
        # Extract using spaCy if available
        if self.nlp:
            spacy_entities = await self._extract_spacy_entities(text, context)
            entities.extend(spacy_entities)
        
        # Extract using patterns
        pattern_entities = await self._extract_pattern_entities(text, context)
        entities.extend(pattern_entities)
        
        # Extract using knowledge base
        kb_entities = await self._extract_knowledge_base_entities(text, context)
        entities.extend(kb_entities)
        
        # Merge and deduplicate entities
        entities = self._merge_entities(entities)
        
        # Score and rank entities
        entities = self._score_entities(entities, text, context)
        
        # Sort by confidence
        entities.sort(key=lambda e: e.confidence, reverse=True)
        
        return entities
    
    async def _extract_spacy_entities(self, text: str, context: str = "") -> List[ExtractedEntity]:
        """Extract entities using spaCy NER."""
        entities = []
        
        if not self.nlp:
            return entities
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                entity_type = self._map_spacy_label(ent.label_)
                if entity_type:
                    entities.append(ExtractedEntity(
                        text=ent.text,
                        entity_type=entity_type,
                        confidence=0.8,  # Base spaCy confidence
                        start_pos=ent.start_char,
                        end_pos=ent.end_char,
                        context=context,
                        properties={"spacy_label": ent.label_}
                    ))
        
        except Exception as e:
            logger.error(f"Error in spaCy entity extraction: {e}")
        
        return entities
    
    def _map_spacy_label(self, label: str) -> Optional[EntityType]:
        """Map spaCy entity labels to our EntityType enum."""
        mapping = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "PRODUCT": EntityType.TECHNOLOGY,
            "EVENT": EntityType.CONCEPT,
            "WORK_OF_ART": EntityType.CONCEPT,
            "LAW": EntityType.CONCEPT,
            "LANGUAGE": EntityType.TECHNOLOGY,
        }
        return mapping.get(label)
    
    async def _extract_pattern_entities(self, text: str, context: str = "") -> List[ExtractedEntity]:
        """Extract entities using regex patterns."""
        entities = []
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                try:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        entity_text = match.group(1) if match.groups() else match.group(0)
                        
                        entities.append(ExtractedEntity(
                            text=entity_text,
                            entity_type=entity_type,
                            confidence=0.7,  # Base pattern confidence
                            start_pos=match.start(),
                            end_pos=match.end(),
                            context=context,
                            properties={"pattern": pattern}
                        ))
                
                except Exception as e:
                    logger.error(f"Error in pattern matching: {e}")
        
        return entities

    async def _extract_knowledge_base_entities(self, text: str, context: str = "") -> List[ExtractedEntity]:
        """Extract entities using knowledge base lookup."""
        entities = []
        text_lower = text.lower()

        # Check for company names
        for company in self.knowledge_base["companies"]:
            if company.lower() in text_lower:
                start_pos = text_lower.find(company.lower())
                entities.append(ExtractedEntity(
                    text=company,
                    entity_type=EntityType.COMPANY,
                    confidence=0.9,
                    start_pos=start_pos,
                    end_pos=start_pos + len(company),
                    context=context,
                    properties={"source": "knowledge_base"}
                ))

        # Check for tech people
        for person in self.knowledge_base["tech_people"]:
            if person.lower() in text_lower:
                start_pos = text_lower.find(person.lower())
                entities.append(ExtractedEntity(
                    text=person,
                    entity_type=EntityType.PERSON,
                    confidence=0.9,
                    start_pos=start_pos,
                    end_pos=start_pos + len(person),
                    context=context,
                    properties={"source": "knowledge_base"}
                ))

        # Check for technology aliases
        for alias, full_name in self.knowledge_base["tech_aliases"].items():
            if alias.lower() in text_lower:
                start_pos = text_lower.find(alias.lower())
                entities.append(ExtractedEntity(
                    text=alias,
                    entity_type=EntityType.TECHNOLOGY,
                    confidence=0.8,
                    start_pos=start_pos,
                    end_pos=start_pos + len(alias),
                    context=context,
                    aliases=[full_name],
                    properties={"source": "knowledge_base", "full_name": full_name}
                ))

        return entities

    def _merge_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Merge overlapping entities, keeping the best ones."""
        if not entities:
            return entities

        # Sort by start position
        entities.sort(key=lambda e: e.start_pos)

        merged = []
        current = entities[0]

        for next_entity in entities[1:]:
            # Check for overlap
            if self._entities_overlap(current, next_entity):
                # Keep the entity with higher confidence or more specific type
                if self._is_better_entity(next_entity, current):
                    current = next_entity
            else:
                merged.append(current)
                current = next_entity

        merged.append(current)
        return merged

    def _entities_overlap(self, entity1: ExtractedEntity, entity2: ExtractedEntity) -> bool:
        """Check if two entities overlap in position."""
        return not (entity1.end_pos <= entity2.start_pos or entity2.end_pos <= entity1.start_pos)

    def _is_better_entity(self, entity1: ExtractedEntity, entity2: ExtractedEntity) -> bool:
        """Determine if entity1 is better than entity2."""
        # Higher confidence wins
        if entity1.confidence != entity2.confidence:
            return entity1.confidence > entity2.confidence

        # More specific entity types win
        specificity_order = {
            EntityType.FUNCTION: 10,
            EntityType.CLASS: 10,
            EntityType.METHOD: 9,
            EntityType.VARIABLE: 8,
            EntityType.FILE: 8,
            EntityType.TECHNOLOGY: 7,
            EntityType.FRAMEWORK: 7,
            EntityType.DATABASE: 7,
            EntityType.COMPANY: 6,
            EntityType.PERSON: 6,
            EntityType.CONCEPT: 5,
            EntityType.ORGANIZATION: 4,
        }

        spec1 = specificity_order.get(entity1.entity_type, 0)
        spec2 = specificity_order.get(entity2.entity_type, 0)

        return spec1 > spec2

    def _score_entities(self, entities: List[ExtractedEntity], text: str, context: str) -> List[ExtractedEntity]:
        """Score entities based on various factors."""
        for entity in entities:
            # Base confidence from extraction method
            base_confidence = entity.confidence

            # Context relevance boost
            context_boost = self._calculate_context_relevance(entity, text, context)

            # Entity type importance boost
            type_boost = self._calculate_type_importance(entity.entity_type)

            # Text position boost (entities at beginning/end might be more important)
            position_boost = self._calculate_position_importance(entity, text)

            # Calculate final confidence
            entity.confidence = min(
                base_confidence + context_boost + type_boost + position_boost,
                1.0
            )

        return entities

    def _calculate_context_relevance(self, entity: ExtractedEntity, text: str, context: str) -> float:
        """Calculate how relevant the entity is to the context."""
        relevance_boost = 0.0

        # Check if entity appears multiple times
        entity_count = text.lower().count(entity.text.lower())
        if entity_count > 1:
            relevance_boost += min(entity_count * 0.05, 0.2)

        # Check context relevance
        if context:
            context_lower = context.lower()
            entity_lower = entity.text.lower()

            if entity_lower in context_lower:
                relevance_boost += 0.1

        return relevance_boost

    def _calculate_type_importance(self, entity_type: EntityType) -> float:
        """Calculate importance boost based on entity type."""
        importance_scores = {
            EntityType.FUNCTION: 0.15,
            EntityType.CLASS: 0.15,
            EntityType.METHOD: 0.12,
            EntityType.TECHNOLOGY: 0.10,
            EntityType.FRAMEWORK: 0.10,
            EntityType.DATABASE: 0.10,
            EntityType.FILE: 0.08,
            EntityType.VARIABLE: 0.05,
            EntityType.COMPANY: 0.08,
            EntityType.PERSON: 0.05,
            EntityType.CONCEPT: 0.03,
        }

        return importance_scores.get(entity_type, 0.0)

    def _calculate_position_importance(self, entity: ExtractedEntity, text: str) -> float:
        """Calculate importance boost based on position in text."""
        text_length = len(text)
        if text_length == 0:
            return 0.0

        # Entities at the beginning get a small boost
        if entity.start_pos < text_length * 0.2:
            return 0.05

        return 0.0

    def get_entity_aliases(self, entity: ExtractedEntity) -> List[str]:
        """Get aliases for an entity."""
        aliases = entity.aliases.copy() if entity.aliases else []

        # Add knowledge base aliases
        if entity.entity_type == EntityType.TECHNOLOGY:
            for alias, full_name in self.knowledge_base["tech_aliases"].items():
                if entity.text.lower() == alias.lower():
                    aliases.append(full_name)
                elif entity.text.lower() == full_name.lower():
                    aliases.append(alias)

        return aliases

    def get_entity_context_keywords(self, entity: ExtractedEntity) -> List[str]:
        """Get context keywords that might be related to this entity."""
        keywords = []

        if entity.entity_type == EntityType.FUNCTION:
            keywords.extend(["method", "function", "call", "invoke", "execute", "run"])
        elif entity.entity_type == EntityType.CLASS:
            keywords.extend(["class", "object", "instance", "inherit", "extend", "implement"])
        elif entity.entity_type == EntityType.DATABASE:
            keywords.extend(["query", "table", "schema", "connection", "transaction", "index"])
        elif entity.entity_type == EntityType.FRAMEWORK:
            keywords.extend(["library", "package", "dependency", "import", "install", "configure"])
        elif entity.entity_type == EntityType.API:
            keywords.extend(["endpoint", "request", "response", "HTTP", "REST", "GraphQL"])

        return keywords
