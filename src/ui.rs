use crate::{
    model::{TensorKind, TensorsRecord},
    tree::{TensorTree, TreeRowKind},
};
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    Frame,
    layout::Constraint,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Row, Table, TableState},
};
pub struct AppState {
    pub records: Vec<TensorsRecord>,
    pub tree: TensorTree,
    pub table_state: TableState,
    pub total_params: usize,
    pub total_memory: usize,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            table_state: TableState::default(),
            records: vec![],
            tree: TensorTree::default(),
            total_params: 0,
            total_memory: 0,
        }
    }
    pub fn count(&mut self) {
        let mut total_params = 0;
        let mut total_memory = 0;
        for r in &self.records {
            total_params += r.numel;
            total_memory += r.size_bytes;
        }
        self.total_memory = total_memory;
        self.total_params = total_params;
    }
    pub fn set_records(&mut self, records: Vec<TensorsRecord>) {
        self.records = records;
        self.tree = TensorTree::from_records(&self.records);
        self.count();
        self.table_state
            .select((!self.records.is_empty()).then_some(0));
    }
}
fn format_bytes(bytes: usize) -> String {
    if bytes >= 1_048_576 {
        format!("{:.2} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

pub fn handle_key_event(app: &mut AppState, key: KeyEvent) {
    let row_count = app.tree.visible_len();
    if row_count == 0 {
        return;
    }
    let selected = app.table_state.selected().unwrap_or(0);
    let new_selected = match key.code {
        KeyCode::Down | KeyCode::Char('j') => (selected + 1) % row_count,
        KeyCode::Up | KeyCode::Char('k') => {
            if selected == 0 {
                row_count - 1
            } else {
                selected - 1
            }
        }
        KeyCode::Enter | KeyCode::Char(' ') | KeyCode::Right => {
            app.tree.toggle_visible_row(selected);
            return;
        }
        _ => return,
    };
    app.table_state.select(Some(new_selected));
}
pub fn draw(frame: &mut Frame, app: &mut AppState) {
    let rows = app.tree.visible_rows().into_iter().map(|tree_row| {
        let indent = "  ".repeat(tree_row.depth);
        match tree_row.kind {
            TreeRowKind::Folder => Row::new(vec![
                Cell::from(Line::from(vec![Span::styled(
                    format!(
                        "{indent}{} {}",
                        if tree_row.expanded { "▼" } else { "▶" },
                        tree_row.name
                    ),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                )])),
                Cell::from("Folder"),
                Cell::from("-"),
                Cell::from(tree_row.numel.to_string()),
                Cell::from(format_bytes(tree_row.size_bytes)),
            ]),
            TreeRowKind::Tensor(record_index) => {
                let record = &app.records[record_index];
                let (kind_str, kind_color) = match record.kind {
                    TensorKind::Weight => ("Weight", Color::Cyan),
                    TensorKind::Bias => ("Bias", Color::Magenta),
                    TensorKind::LayerNorm => ("LayerNorm", Color::Yellow),
                    TensorKind::Attention => ("Attention", Color::Green),
                    TensorKind::Embedding => ("Embedding", Color::Blue),
                    TensorKind::Other => ("Other", Color::Gray),
                };
                Row::new(vec![
                    Cell::from(Line::from(vec![Span::styled(
                        format!("{indent}  {}", tree_row.name),
                        Style::default().fg(Color::White),
                    )])),
                    Cell::from(Span::styled(kind_str, Style::default().fg(kind_color))),
                    Cell::from(format!("{:?}", record.shape)),
                    Cell::from(record.numel.to_string()),
                    Cell::from(format_bytes(record.size_bytes)),
                ])
            }
        }
    });
    let widths = vec![
        Constraint::Percentage(40),
        Constraint::Percentage(15),
        Constraint::Percentage(20),
        Constraint::Percentage(10),
        Constraint::Percentage(15),
    ];
    let header = Row::new(vec!["Tensor Path", "Kind", "Shape", "Params", "Size"]).style(
        Style::default()
            .add_modifier(Modifier::BOLD)
            .add_modifier(Modifier::UNDERLINED),
    );
    let table = Table::new(rows, widths)
        .header(header)
        .block(
            Block::default()
                .title("Model Weight  (j/k: move, Enter/Space/→: toggle, q: quit)")
                .borders(Borders::ALL),
        )
        .row_highlight_style(Style::default().bg(Color::DarkGray));
    frame.render_stateful_widget(table, frame.area(), &mut app.table_state);
}
