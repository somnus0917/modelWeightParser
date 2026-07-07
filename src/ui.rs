use ratatui::{
    Frame,
    widgets::{Block, Borders, Paragraph},
};

pub fn draw(frame: &mut Frame) {
    let text =
        Paragraph::new("模块化重构成功！\nUI 逻辑现在独立在这个文件里了。\n\n按 'q' 键退出。")
            .block(Block::default().title(" UI 模块 ").borders(Borders::ALL));
    frame.render_widget(text, frame.area());
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};

    #[test]
    fn test_ui_draw() {
        let backend = TestBackend::new(20, 3);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| {
                draw(f);
            })
            .unwrap();
    }
}
