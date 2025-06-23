# Chat Interface Project Filtering

The Chat Interface now supports **project-specific knowledge filtering**, allowing users to select one or multiple projects for focused conversations. This ensures that chat responses only include knowledge from the selected projects, excluding any unselected projects from the search results.

## 🎯 Key Features

### ✅ Project Selection
- **Multi-project selection**: Choose one or multiple projects using checkboxes
- **Select All/None**: Quick toggle for all indexed projects
- **Visual indicators**: Clear status indicators for each project (indexed/indexing/error)
- **Real-time updates**: Project list updates automatically as projects are indexed

### ✅ Persistent State
- **localStorage persistence**: Selected projects are saved automatically
- **Cross-session persistence**: Selection survives browser restarts
- **Page refresh resilience**: Selection maintained when refreshing the page
- **Clear selection**: Easy reset button to clear all selections

### ✅ Visual Feedback
- **Empty state indicators**: Clear messaging about current project selection
- **Input area indicators**: Small project count display in chat input
- **Settings panel**: Prominent project selection UI with helpful tips
- **Status warnings**: Visual alerts when no projects are selected

## 🔧 Technical Implementation

### Frontend (React)
- **Custom hook**: `usePersistedProjectSelection()` for state management
- **localStorage integration**: Automatic save/load of project selections
- **Enhanced UI**: Improved project selector with better visual design
- **Error handling**: Graceful handling of localStorage errors

### Backend (FastAPI)
- **Project filtering**: Vector store and graph store filtering by `project_ids`
- **API parameter**: `project_ids` array in query requests
- **Database filtering**: Qdrant and Neo4j queries scoped to selected projects
- **Context retrieval**: Related chunks limited to selected projects only

## 📋 Usage Examples

### 1. Select All Projects (Default)
```json
{
  "query": "find authentication functions",
  "project_ids": null
}
```
**Result**: Searches across all indexed projects

### 2. Select Specific Project
```json
{
  "query": "show me error handling",
  "project_ids": ["28bdd67c-bfbe-452d-9853-bab264103cda"]
}
```
**Result**: Only searches within the selected project

### 3. Select Multiple Projects
```json
{
  "query": "find database connections",
  "project_ids": [
    "28bdd67c-bfbe-452d-9853-bab264103cda",
    "22b81f02-ce85-4573-9b16-fc936a4a8ca5"
  ]
}
```
**Result**: Searches only within the 2 selected projects

## 🎨 User Interface

### Project Selection Panel
Located in the chat settings (gear icon), the project selection panel features:

- **Highlighted section**: Blue-bordered panel with clear labeling
- **Dropdown interface**: Clean project selector with checkboxes
- **Status indicators**: Visual project status (✅ indexed, 🔄 indexing, ❌ error)
- **Selection summary**: Real-time count of selected projects
- **Clear button**: Quick reset for project selection
- **Helpful tips**: Guidance text explaining the functionality

### Chat Interface Indicators
- **Empty state**: Shows current project selection status
- **Input area**: Small indicator showing number of selected projects
- **Warning messages**: Alerts when no projects are selected

## 🔍 Search Behavior

### When Projects Are Selected
- ✅ **Vector search**: Limited to chunks from selected projects only
- ✅ **Graph traversal**: Context retrieval scoped to selected projects
- ✅ **Embedding search**: Only embeddings from selected projects considered
- ✅ **Result filtering**: All results guaranteed to be from selected projects

### When No Projects Are Selected
- ⚠️ **Fallback behavior**: Searches across all indexed projects
- ⚠️ **Visual warning**: UI shows warning about searching all projects
- ⚠️ **Recommendation**: Suggests selecting specific projects for focused results

## 💾 Data Persistence

### localStorage Schema
```javascript
// Key: 'chatInterface_selectedProjects'
// Value: JSON array of project IDs
["project-id-1", "project-id-2", "project-id-3"]
```

### Error Handling
- **Parse errors**: Gracefully handles corrupted localStorage data
- **Invalid data**: Automatically clears invalid project selections
- **Missing projects**: Handles cases where saved projects no longer exist

## 🚀 Benefits

### For Users
- **Focused conversations**: Chat responses relevant to specific projects only
- **Reduced noise**: Eliminates irrelevant results from other projects
- **Better context**: More accurate code analysis within project scope
- **Persistent workflow**: Maintains project focus across sessions

### For Developers
- **Modular architecture**: Clean separation between UI and backend filtering
- **Scalable design**: Efficient filtering at database level
- **Maintainable code**: Custom hooks and reusable components
- **Testable logic**: Clear API contracts and isolated functionality

## 🧪 Testing

Run the demo script to see project filtering in action:

```bash
cd backend
python demo_project_filtering.py
```

This will show:
- Available indexed projects
- Example API requests with project filtering
- Expected behavior for different selection scenarios

## 🔮 Future Enhancements

- **Project groups**: Organize projects into logical groups
- **Recent projects**: Quick access to recently used projects
- **Project search**: Filter project list by name or description
- **Bulk operations**: Select projects by criteria (e.g., all Python projects)
- **Project favorites**: Mark frequently used projects for quick access
