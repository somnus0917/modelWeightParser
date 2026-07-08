use crate::model::{TensorKind, TensorsRecord};
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
    pub table_state: TableState,
    pub total_params: usize,
    pub total_memory: usize,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            table_state: TableState::default(),
            records: vec![],
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
    pub fn add_record(&mut self, record: TensorsRecord) {
        self.records.push(record);
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
    let row_count = app.records.len();
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
        _ => return,
    };
    app.table_state.select(Some(new_selected));
}
pub fn draw(frame: &mut Frame, records: &[TensorsRecord], table_state: &mut TableState) {
    let rows = records.iter().map(|record| {
        let mut name_spans = Vec::new();
        let path_len = record.module_path.len();
        for (i, part) in record.module_path.iter().enumerate() {
            if i == path_len - 1 {
                name_spans.push(Span::styled(
                    part,
                    Style::default()
                        .fg(ratatui::style::Color::White)
                        .add_modifier(Modifier::BOLD),
                ));
            } else {
                name_spans.push(Span::styled(
                    format!("{}", part),
                    Style::default().fg(ratatui::style::Color::DarkGray),
                ));
            }
        }
        let (kind_str, kind_color) = match record.kind {
            TensorKind::Weight => ("Weight", Color::Cyan),
            TensorKind::Bias => ("Bias", Color::Magenta),
            TensorKind::LayerNorm => ("LayerNorm", Color::Yellow),
            TensorKind::Attention => ("Attention", Color::Green),
            TensorKind::Embedding => ("Embedding", Color::Blue),
            TensorKind::Other => ("Other", Color::Gray),
        };
        Row::new(vec![
            Cell::from(Line::from(name_spans)),
            Cell::from(Span::styled(kind_str, Style::default().fg(kind_color))),
            Cell::from(format!("{:?}", record.shape)),
            Cell::from(format_bytes(record.numel)),
            Cell::from(format_bytes(record.size_bytes)),
        ])
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
        .block(Block::default().title("Model Weight").borders(Borders::ALL))
        .row_highlight_style(Style::default().bg(Color::DarkGray));
    frame.render_stateful_widget(table, frame.area(), table_state);
}
