use std::{cmp::Ordering, collections::BTreeMap};

use crate::model::TensorsRecord;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeRowKind {
    Folder,
    Tensor(usize),
}

#[derive(Debug, Clone)]
pub struct TreeRow {
    pub name: String,
    pub depth: usize,
    pub kind: TreeRowKind,
    pub expanded: bool,
    pub numel: usize,
    pub size_bytes: usize,
}

#[derive(Debug, Default)]
struct TreeNode {
    name: String,
    children: BTreeMap<String, TreeNode>,
    record_index: Option<usize>,
    expanded: bool,
    numel: usize,
    size_bytes: usize,
}

fn natural_name_cmp(left: &str, right: &str) -> Ordering {
    match (left.parse::<u64>(), right.parse::<u64>()) {
        (Ok(left_number), Ok(right_number)) => {
            left_number.cmp(&right_number).then_with(|| left.cmp(right))
        }
        _ => left.cmp(right),
    }
}

impl TreeNode {
    fn sorted_child_names(&self) -> Vec<&String> {
        let mut names: Vec<_> = self.children.keys().collect();
        names.sort_by(|left, right| natural_name_cmp(left, right));
        names
    }

    fn insert(&mut self, path: &[String], record_index: usize, record: &TensorsRecord) {
        self.numel += record.numel;
        self.size_bytes += record.size_bytes;

        let Some((segment, rest)) = path.split_first() else {
            self.record_index = Some(record_index);
            return;
        };

        let child = self
            .children
            .entry(segment.clone())
            .or_insert_with(|| TreeNode {
                name: segment.clone(),
                ..Self::default()
            });
        child.insert(rest, record_index, record);
    }

    fn visible_rows(&self, depth: usize, rows: &mut Vec<TreeRow>) {
        for name in self.sorted_child_names() {
            let child = &self.children[name];
            let is_folder = !child.children.is_empty();
            rows.push(TreeRow {
                name: child.name.clone(),
                depth,
                kind: if is_folder {
                    TreeRowKind::Folder
                } else {
                    TreeRowKind::Tensor(child.record_index.expect("leaf nodes have a record"))
                },
                expanded: child.expanded,
                numel: child.numel,
                size_bytes: child.size_bytes,
            });

            if is_folder && child.expanded {
                child.visible_rows(depth + 1, rows);
            }
        }
    }

    fn toggle_visible_row(&mut self, target: usize, cursor: &mut usize) -> bool {
        let mut names: Vec<_> = self.children.keys().cloned().collect();
        names.sort_by(|left, right| natural_name_cmp(left, right));

        for name in names {
            let child = self
                .children
                .get_mut(&name)
                .expect("child name came from this node");
            let is_folder = !child.children.is_empty();
            if *cursor == target {
                if is_folder {
                    child.expanded = !child.expanded;
                    return true;
                }
                return false;
            }
            *cursor += 1;

            if is_folder && child.expanded && child.toggle_visible_row(target, cursor) {
                return true;
            }
        }
        false
    }
}

#[derive(Debug, Default)]
pub struct TensorTree {
    root: TreeNode,
}

impl TensorTree {
    pub fn from_records(records: &[TensorsRecord]) -> Self {
        let mut tree = Self::default();
        for (index, record) in records.iter().enumerate() {
            tree.root.insert(&record.module_path, index, record);
        }
        tree
    }

    pub fn visible_rows(&self) -> Vec<TreeRow> {
        let mut rows = Vec::new();
        self.root.visible_rows(0, &mut rows);
        rows
    }

    pub fn visible_len(&self) -> usize {
        self.visible_rows().len()
    }

    pub fn toggle_visible_row(&mut self, row_index: usize) -> bool {
        self.root.toggle_visible_row(row_index, &mut 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{TensorKind, TensorsRecord};

    fn record(name: &str) -> TensorsRecord {
        TensorsRecord {
            name: name.into(),
            dtype: "F32".into(),
            shape: vec![1],
            numel: 1,
            size_bytes: 4,
            module_path: name.split('.').map(str::to_owned).collect(),
            kind: TensorKind::Weight,
        }
    }

    #[test]
    fn numeric_folder_names_are_sorted_numerically() {
        let records = vec![
            record("encoder.layer.10.weight"),
            record("encoder.layer.2.weight"),
            record("encoder.layer.0.weight"),
            record("encoder.layer.11.weight"),
            record("encoder.layer.1.weight"),
        ];
        let mut tree = TensorTree::from_records(&records);

        assert!(tree.toggle_visible_row(0)); // encoder
        assert!(tree.toggle_visible_row(1)); // layer
        let names: Vec<_> = tree
            .visible_rows()
            .into_iter()
            .skip(2)
            .map(|row| row.name)
            .collect();
        assert_eq!(names, ["0", "1", "2", "10", "11"]);
    }

    #[test]
    fn builds_top_level_folders_and_only_shows_expanded_children() {
        let records = vec![
            record("encoder.layer.0.weight"),
            record("embeddings.word.weight"),
        ];
        let mut tree = TensorTree::from_records(&records);

        let rows = tree.visible_rows();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].name, "embeddings");
        assert_eq!(rows[1].name, "encoder");
        assert!(matches!(rows[1].kind, TreeRowKind::Folder));
        assert_eq!(rows[1].numel, 1);

        assert!(tree.toggle_visible_row(1));
        let rows = tree.visible_rows();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[2].name, "layer");
        assert_eq!(rows[2].depth, 1);
    }
}
