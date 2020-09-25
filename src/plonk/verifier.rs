use super::{
    circuit::{default_query_set, inv_query_set},
    hash_point, Error, Proof, SRS,
};
use crate::arithmetic::{get_challenge_scalar, Challenge, CurveAffine, Field};
use crate::poly::commitment::{Guard, Params, MSM};
use crate::transcript::Hasher;

impl<'a, C: CurveAffine> Proof<C> {
    /// Returns a boolean indicating whether or not the proof is valid
    pub fn verify<HBase: Hasher<C::Base>, HScalar: Hasher<C::Scalar>>(
        &self,
        params: &'a Params<C>,
        srs: &SRS<C>,
        mut msm: MSM<'a, C>,
        aux_commitments: &[C],
    ) -> Result<Guard<'a, C>, Error> {
        // Check that aux_commitments matches the expected number of aux_wires
        // and self.aux_evals
        if aux_commitments.len() != srs.cs.num_aux_wires
            || self.aux_evals.len() != srs.cs.num_aux_wires
        {
            return Err(Error::IncompatibleParams);
        }

        // Scale the MSM by a random factor to ensure that if the existing MSM
        // has is_zero() == false then this argument won't be able to interfere
        // with it to make it true, with high probability.
        msm.scale(C::Scalar::random());

        // Create a transcript for obtaining Fiat-Shamir challenges.
        let mut transcript = HBase::init(C::Base::one());

        // Hash the aux (external) commitments into the transcript
        for commitment in aux_commitments {
            hash_point(&mut transcript, commitment)?;
        }

        // Hash the prover's advice commitments into the transcript
        for commitment in &self.advice_commitments {
            hash_point(&mut transcript, commitment)?;
        }

        // Sample x_0 challenge
        let x_0: C::Scalar = get_challenge_scalar(Challenge(transcript.squeeze().get_lower_128()));

        // Sample x_1 challenge
        let x_1: C::Scalar = get_challenge_scalar(Challenge(transcript.squeeze().get_lower_128()));

        // Hash each permutation product commitment
        for c in &self.permutation_product_commitments {
            hash_point(&mut transcript, c)?;
        }

        // Sample x_2 challenge, which keeps the gates linearly independent.
        let x_2: C::Scalar = get_challenge_scalar(Challenge(transcript.squeeze().get_lower_128()));

        // Obtain a commitment to h(X) in the form of multiple pieces of degree n - 1
        for c in &self.h_commitments {
            hash_point(&mut transcript, c)?;
        }

        // Sample x_3 challenge, which is used to ensure the circuit is
        // satisfied with high probability.
        let x_3: C::Scalar = get_challenge_scalar(Challenge(transcript.squeeze().get_lower_128()));
        let x_3n = x_3.pow(&[params.n as u64, 0, 0, 0]);

        // Hash together all the openings provided by the prover into a new
        // transcript on the scalar field.
        let mut transcript_scalar = HScalar::init(C::Scalar::one());

        for eval in self
            .advice_evals
            .iter()
            .chain(self.aux_evals.iter())
            .chain(self.fixed_evals.iter())
            .chain(self.h_evals.iter())
            .chain(self.permutation_product_evals.iter())
            .chain(self.permutation_product_inv_evals.iter())
            .chain(self.permutation_evals.iter().flat_map(|evals| evals.iter()))
        {
            transcript_scalar.absorb(*eval);
        }

        let transcript_scalar_point =
            C::Base::from_bytes(&(transcript_scalar.squeeze()).to_bytes()).unwrap();
        transcript.absorb(transcript_scalar_point);

        // Evaluate the circuit using the custom gates provided
        let mut h_eval = C::Scalar::zero();
        for poly in srs.cs.gates.iter() {
            h_eval *= &x_2;

            let evaluation: C::Scalar = poly.evaluate(
                &|index| self.fixed_evals[index],
                &|index| self.advice_evals[index],
                &|index| self.aux_evals[index],
                &|a, b| a + &b,
                &|a, b| a * &b,
                &|a, scalar| a * &scalar,
            );

            h_eval += &evaluation;
        }

        // First element in each permutation product should be 1
        // l_0(X) * (1 - z(X)) = 0
        {
            // TODO: bubble this error up
            let denominator = (x_3 - &C::Scalar::one()).invert().unwrap();

            for eval in self.permutation_product_evals.iter() {
                h_eval *= &x_2;

                let mut tmp = denominator; // 1 / (x_3 - 1)
                tmp *= &(x_3n - &C::Scalar::one()); // (x_3^n - 1) / (x_3 - 1)
                tmp *= &srs.domain.get_barycentric_weight(); // l_0(x_3)
                tmp *= &(C::Scalar::one() - &eval); // l_0(X) * (1 - z(X))

                h_eval += &tmp;
            }
        }

        // z(X) \prod (p(X) + \beta s_i(X) + \gamma) - z(omega^{-1} X) \prod (p(X) + \delta^i \beta X + \gamma)
        for (permutation_index, wires) in srs.cs.permutations.iter().enumerate() {
            h_eval *= &x_2;

            let mut left = self.permutation_product_evals[permutation_index];
            for (advice_eval, permutation_eval) in wires
                .iter()
                .map(|&(_, query_index)| self.advice_evals[query_index])
                .zip(self.permutation_evals[permutation_index].iter())
            {
                left *= &(advice_eval + &(x_0 * permutation_eval) + &x_1);
            }

            let mut right = self.permutation_product_inv_evals[permutation_index];
            let mut current_delta = x_0 * &x_3;
            for advice_eval in wires
                .iter()
                .map(|&(_, query_index)| self.advice_evals[query_index])
            {
                right *= &(advice_eval + &current_delta + &x_1);
                current_delta *= &C::Scalar::DELTA;
            }

            h_eval += &left;
            h_eval -= &right;
        }

        // Compute the expected h(x) value
        let mut expected_h_eval = C::Scalar::zero();
        let mut cur = C::Scalar::one();
        for eval in &self.h_evals {
            expected_h_eval += &(cur * eval);
            cur *= &x_3n;
        }

        if h_eval != (expected_h_eval * &(x_3n - &C::Scalar::one())) {
            return Err(Error::ConstraintSystemFailure);
        }

        // We are now convinced the circuit is satisfied so long as the
        // polynomial commitments open to the correct values.

        // Sample x_4 for compressing openings at the same points together
        let x_4: C::Scalar = get_challenge_scalar(Challenge(transcript.squeeze().get_lower_128()));

        // Compress the commitments and expected evaluations at x_3 together
        // using the challenge x_4
        let mut q_commitments: Vec<_> = vec![params.empty_msm(); srs.cs.query_sets.len()];
        let mut q_eval_sets: Vec<Vec<C::Scalar>> = vec![Vec::new(); srs.cs.query_sets.len()];
        for (set_idx, query_set) in srs.cs.query_sets.iter().enumerate() {
            q_eval_sets[set_idx] = vec![C::Scalar::zero(); query_set.len()];
        }

        {
            let mut accumulate = |set_idx: usize, new_commitment, evals: Vec<C::Scalar>| {
                q_commitments[set_idx].scale(x_4);
                q_commitments[set_idx].add_term(C::Scalar::one(), new_commitment);
                for query_idx in 0..q_eval_sets[set_idx].len() {
                    q_eval_sets[set_idx][query_idx] *= &x_4;
                    q_eval_sets[set_idx][query_idx] += &evals[query_idx];
                }
            };

            for (set_idx, query_set) in srs.cs.query_sets.iter().enumerate() {
                // If advice wires were queried at this set
                if let Some(advice_wires) = srs.cs.advice_query_sets.clone().get_mut(query_set) {
                    for wire in advice_wires.iter() {
                        let mut wire_evals = vec![C::Scalar::zero(); query_set.len()];
                        for (query_idx, query) in query_set.iter().enumerate() {
                            let advice_query_idx = srs
                                .cs
                                .query_existing_advice_index(*wire, (*query).0)
                                .unwrap();
                            wire_evals[query_idx] = self.advice_evals[advice_query_idx];
                        }
                        accumulate(set_idx, self.advice_commitments[wire.0], wire_evals);
                    }
                }

                // If aux wires were queried at this set
                if let Some(aux_wires) = srs.cs.aux_query_sets.clone().get_mut(query_set) {
                    for wire in aux_wires.iter() {
                        let mut wire_evals = vec![C::Scalar::zero(); query_set.len()];
                        for (query_idx, query) in query_set.iter().enumerate() {
                            let aux_query_idx =
                                srs.cs.query_existing_aux_index(*wire, (*query).0).unwrap();
                            wire_evals[query_idx] = self.aux_evals[aux_query_idx];
                        }
                        accumulate(set_idx, aux_commitments[wire.0], wire_evals);
                    }
                }

                // If fixed wires were queried at this set
                if let Some(fixed_wires) = srs.cs.fixed_query_sets.clone().get_mut(query_set) {
                    for wire in fixed_wires.iter() {
                        let mut wire_evals = vec![C::Scalar::zero(); query_set.len()];
                        for (query_idx, query) in query_set.iter().enumerate() {
                            let fixed_query_idx = srs
                                .cs
                                .query_existing_fixed_index(*wire, (*query).0)
                                .unwrap();
                            wire_evals[query_idx] = self.fixed_evals[fixed_query_idx];
                        }
                        accumulate(set_idx, srs.fixed_commitments[wire.0], wire_evals);
                    }
                }
            }

            let default_set_index = srs
                .cs
                .query_sets
                .iter()
                .position(|set| set == &default_query_set())
                .unwrap();
            let inv_set_index = srs
                .cs
                .query_sets
                .iter()
                .position(|set| set == &inv_query_set())
                .unwrap();

            for (commitment, eval) in self.h_commitments.iter().zip(self.h_evals.iter()) {
                accumulate(default_set_index, *commitment, vec![*eval]);
            }

            // Handle permutation arguments, if any exist
            if !srs.cs.permutations.is_empty() {
                // Open permutation product commitments at x_3
                for (commitment, eval) in self
                    .permutation_product_commitments
                    .iter()
                    .zip(self.permutation_product_evals.iter())
                {
                    accumulate(default_set_index, *commitment, vec![*eval]);
                }
                // Open permutation commitments for each permutation argument at x_3
                for (commitment, eval) in srs
                    .permutation_commitments
                    .iter()
                    .zip(self.permutation_evals.iter())
                    .flat_map(|(commitments, evals)| commitments.iter().zip(evals.iter()))
                {
                    accumulate(default_set_index, *commitment, vec![*eval]);
                }
                // Open permutation product commitments at \omega^{-1} x_3
                for (commitment, eval) in self
                    .permutation_product_commitments
                    .iter()
                    .zip(self.permutation_product_inv_evals.iter())
                {
                    accumulate(inv_set_index, *commitment, vec![*eval]);
                }
            }
        }

        // Sample a challenge x_5 for keeping the multi-point quotient
        // polynomial terms linearly independent.
        let x_5: C::Scalar = get_challenge_scalar(Challenge(transcript.squeeze().get_lower_128()));

        // Obtain the commitment to the multi-point quotient polynomial f(X).
        hash_point(&mut transcript, &self.f_commitment)?;

        // Sample a challenge x_6 for checking that f(X) was committed to
        // correctly.
        let x_6: C::Scalar = get_challenge_scalar(Challenge(transcript.squeeze().get_lower_128()));

        for eval in self.q_evals.iter() {
            transcript_scalar.absorb(*eval);
        }

        let transcript_scalar_point =
            C::Base::from_bytes(&(transcript_scalar.squeeze()).to_bytes()).unwrap();
        transcript.absorb(transcript_scalar_point);

        // We can compute the expected msm_eval at x_6 using the q_evals provided
        // by the prover and from x_5
        let mut msm_eval = C::Scalar::zero();
        for (set_idx, query_set) in srs.cs.query_sets.iter().enumerate() {
            let mut eval = self.q_evals[set_idx];
            for (query_idx, &row) in query_set.iter().enumerate() {
                eval = eval - &q_eval_sets[set_idx][query_idx];
                let point = srs.domain.rotate_omega(x_3, row);
                eval = eval * &(x_6 - &point).invert().unwrap();
                msm_eval *= &x_5;
                msm_eval += &eval;
            }
        }

        // Sample a challenge x_7 that we will use to collapse the openings of
        // the various remaining polynomials at x_6 together.
        let x_7: C::Scalar = get_challenge_scalar(Challenge(transcript.squeeze().get_lower_128()));

        // Compute the final commitment that has to be opened
        let mut commitment_msm = params.empty_msm();
        commitment_msm.add_term(C::Scalar::one(), self.f_commitment);
        for (set_idx, _) in srs.cs.query_sets.iter().enumerate() {
            commitment_msm.scale(x_7);
            commitment_msm.add_msm(&q_commitments[set_idx]);
            msm_eval *= &x_7;
            msm_eval += &self.q_evals[set_idx];
        }

        // Verify the opening proof
        self.opening
            .verify(params, msm, &mut transcript, x_6, commitment_msm, msm_eval)
            .map_err(|_| Error::OpeningError)
    }
}
