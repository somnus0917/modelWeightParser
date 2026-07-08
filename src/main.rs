use crossterm::{
    ExecutableCommand,
    event::{self, Event, KeyCode, KeyEvent},
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use model_weight_parser::model::{TensorKind, TensorsRecord};
use model_weight_parser::ui;
use ratatui::{Terminal, backend::CrosstermBackend};
use std::io::{Result, stdout};
fn main() -> Result<()> {
    enable_raw_mode()?;
    // 进入备用屏幕（类似打开 vim，退出后终端恢复你原来的命令行历史）
    stdout().execute(EnterAlternateScreen)?;
    let record1 = TensorsRecord {
        name: String::from(
            "albert.encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight",
        ),
        dtype: String::from("F32"),
        shape: vec![768, 768],
        numel: 589_824,        // 768 * 768
        size_bytes: 2_359_296, // 589_824 * 4 (F32 占 4 字节)
        module_path: vec![
            String::from("albert"),
            String::from("encoder"),
            String::from("albert_layer_groups"),
            String::from("0"),
            String::from("albert_layers"),
            String::from("0"),
            String::from("attention"),
            String::from("query"),
        ],
        kind: TensorKind::Weight, // 或者根据你的分类逻辑，归为 TensorKind::Attention
    };
    let record2 = TensorsRecord {
        name: String::from("albert.embeddings.LayerNorm.bias"),
        dtype: String::from("F32"),
        shape: vec![128],
        numel: 128,
        size_bytes: 512, // 128 * 4
        module_path: vec![
            String::from("albert"),
            String::from("embeddings"),
            String::from("LayerNorm"),
        ],
        kind: TensorKind::Bias,
    };
    let mut app = ui::AppState::new();
    app.add_record(record1);
    app.add_record(record2);
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;
    loop {
        terminal.draw(|frame| {
            ui::draw(frame, &app.records, &mut app.table_state);
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

    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;

    Ok(())
}
