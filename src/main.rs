use anyhow::Result;
use crossterm::{
    ExecutableCommand,
    event::{self, Event, KeyCode},
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use model_weight_parser::model::{self};
use model_weight_parser::ui;
use ratatui::{Terminal, backend::CrosstermBackend};
use std::io::stdout;

struct TerminalGuard;
impl TerminalGuard {
    fn new() -> std::io::Result<Self> {
        enable_raw_mode()?;
        stdout().execute(EnterAlternateScreen)?;
        Ok(Self)
    }
}
impl Drop for TerminalGuard {
    fn drop(&mut self) {
        disable_raw_mode().ok();
        stdout().execute(LeaveAlternateScreen).ok();
    }
}
fn main() -> Result<()> {
    let _guard = TerminalGuard::new()?;
    let mut app = ui::AppState::new();
    let path = "hf-downloads/all-MiniLM-L12-v2/model.safetensors";
    let records = model::load_safetensors(path)?;
    app.set_records(records);
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;
    loop {
        terminal.draw(|frame| {
            ui::draw(frame, &mut app);
        })?;
        if event::poll(std::time::Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.code == KeyCode::Char('q') {
                    break;
                }
                ui::handle_key_event(&mut app, key);
            }
        }
    }

    Ok(())
}
