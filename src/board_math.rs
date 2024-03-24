#[derive(Clone)]
pub struct BoardMath {
    pub mirror_2d: Vec<Vec<(usize, usize)>>,
    pub l90_2d: Vec<Vec<(usize, usize)>>,
    pub l90_2d_mirror: Vec<Vec<(usize, usize)>>,
    pub r90_2d: Vec<Vec<(usize, usize)>>,
    pub r90_2d_mirror: Vec<Vec<(usize, usize)>>,
    pub l180_2d: Vec<Vec<(usize, usize)>>,
    pub l180_2d_mirror: Vec<Vec<(usize, usize)>>,
    pub mirror_1d: Vec<usize>,
    pub l90_1d: Vec<usize>,
    pub l90_1d_mirror: Vec<usize>,
    pub r90_1d: Vec<usize>,
    pub r90_1d_mirror: Vec<usize>,
    pub l180_1d: Vec<usize>,
    pub l180_1d_mirror: Vec<usize>,
}

impl BoardMath {
    pub fn new(n: usize) -> Self {
        let coords: Vec<Vec<(usize, usize)>> = Self::coordinates(n);

        // mirror (reverse rows)
        let mut mirror_2d = coords.clone();
        for row in &mut mirror_2d {
            row.reverse();
        }

        // left rotation 90 degrees 2D array (transpose -> reverse rows)
        let mut l90_2d = Self::transpose2(coords.clone());
        let l90_2d_mirror = l90_2d.clone();
        for row in &mut l90_2d {
            row.reverse();
        }

        // right rotation 90 degrees 2D array (transpose -> reverse columns)
        let mut r90_2d = Self::transpose2(coords.clone());
        r90_2d.reverse();
        let mut r90_2d_mirror = r90_2d.clone();
        for row in &mut r90_2d_mirror {
            row.reverse();
        }

        // left rotation 180 degrees 2D array (reverse rows -> reverse columns)
        let mut l180_2d = coords.clone();
        l180_2d.reverse();
        let l180_2d_mirror = l180_2d.clone();
        for row in &mut l180_2d {
            row.reverse();
        }

        let mirror_1d = Self::flatten(&mirror_2d);
        let l90_1d = Self::flatten(&l90_2d);
        let l90_1d_mirror = Self::flatten(&l90_2d_mirror);
        let r90_1d = Self::flatten(&r90_2d);
        let r90_1d_mirror = Self::flatten(&r90_2d_mirror);
        let l180_1d = Self::flatten(&l180_2d);
        let l180_1d_mirror = Self::flatten(&l180_2d_mirror);

        // create class
        let b = BoardMath {
            mirror_2d,
            l90_2d,
            l90_2d_mirror,
            r90_2d,
            r90_2d_mirror,
            l180_2d,
            l180_2d_mirror,
            mirror_1d,
            l90_1d,
            l90_1d_mirror,
            r90_1d,
            r90_1d_mirror,
            l180_1d,
            l180_1d_mirror,
        };
        b
    }

    pub fn apply_2d(b: &Vec<Vec<i8>>, t: &Vec<Vec<(usize, usize)>>) -> Vec<Vec<i8>> {
        let mut r = b.clone();
        for x in 0..b.len() {
            for y in 0..b[x].len() {
                r[x][y] = b[t[x][y].0][t[x][y].1];
            }
        }
        r
    }

    pub fn apply_1d(b: &Vec<f32>, t: &Vec<usize>) -> Vec<f32> {
        let mut r = b.clone();
        for x in 0..t.len() {
            r[x] = b[t[x]];
        }
        r
    }

    fn flatten(b_2d: &Vec<Vec<(usize, usize)>>) -> Vec<usize> {
        let dim = b_2d.len();
        let mut b_1d = Vec::with_capacity(b_2d.len() * b_2d.len());
        unsafe { b_1d.set_len(b_2d.len() * b_2d.len()) }
        for x in 0..b_2d.len() {
            for y in 0..b_2d.len() {
                b_1d[x * dim + y] = b_2d[x][y].0 * dim + b_2d[x][y].1;
            }
        }
        b_1d
    }

    fn coordinates(n: usize) -> Vec<Vec<(usize, usize)>> {
        let mut coords: Vec<Vec<(usize, usize)>> = Vec::with_capacity(n);
        for x in 0..n {
            coords.push(Vec::with_capacity(n));
            for y in 0..n {
                coords[x].push((x, y));
            }
        }
        coords
    }

    fn transpose2<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
        assert!(!v.is_empty());
        let len = v[0].len();
        let mut iters: Vec<_> = v.into_iter().map(|n| n.into_iter()).collect();
        (0..len)
            .map(|_| {
                iters
                    .iter_mut()
                    .map(|n| n.next().unwrap())
                    .collect::<Vec<T>>()
            })
            .collect()
    }
}
