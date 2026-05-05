WITH IA_Combined AS (
    SELECT 
        C.InvestmentID,
        C.InvestorID,
        C.EffectiveDate,
        C.ContributionType AS TypeID,
        (C.Amount * -1) AS Amt,
        S.SubtypeUID,
        S.MajorType,
        S.Typename,
        CASE 
            WHEN C.InvestorID LIKE 'OP%' THEN 'Operating Partner'
            ELSE 'Preferred Equity'
        END AS Partner,
        (SELECT SUM(Amt)
         FROM (
             SELECT (C2.Amount * -1) AS Amt
             FROM IA_Contribution C2
             LEFT JOIN IA_Subtype S2 ON C2.ContributionType = S2.SubtypeUID
             WHERE C2.InvestmentID = C.InvestmentID
               AND C2.InvestorID = C.InvestorID
               AND (S2.Typename = 'Contribution: Investments' OR S2.Typename = 'Distribution: Return of Capital')
               AND C2.EffectiveDate <= C.EffectiveDate
             
             UNION ALL
             
             SELECT D2.Amount AS Amt
             FROM IA_Distribution D2
             LEFT JOIN IA_Subtype S2 ON D2.DistributionTypeID = S2.SubtypeUID
             WHERE D2.InvestmentID = C.InvestmentID
               AND D2.InvestorID = C.InvestorID
               AND S2.Typename = 'Distribution: Return of Capital'
               AND D2.EffectiveDate <= C.EffectiveDate
         ) AS Combined
        ) AS Cum_Amt,
        CASE
            WHEN S.MajorType = 'Contribution' THEN 'Y'
            ELSE 'N'
        END AS Capital,
        CASE
            WHEN S.MajorType = 'Contribution' THEN 'N'
            ELSE 'N'
        END AS ROE_Income
    FROM 
        IA_Contribution C
    LEFT JOIN 
        IA_Subtype S 
        ON C.ContributionType = S.SubtypeUID
        AND S.MajorType = 'Contribution'
    
    UNION ALL
    
    SELECT 
        D.InvestmentID,
        D.InvestorID,
        D.EffectiveDate,
        D.DistributionTypeID AS TypeID,
        D.Amount AS Amt,
        S.SubtypeUID,
        S.MajorType,
        S.Typename,
        CASE 
            WHEN D.InvestorID LIKE 'OP%' THEN 'Operating Partner'
            ELSE 'Preferred Equity'
        END AS Partner,
        (SELECT SUM(Amt)
         FROM (
             SELECT (C2.Amount * -1) AS Amt
             FROM IA_Contribution C2
             LEFT JOIN IA_Subtype S2 ON C2.ContributionType = S2.SubtypeUID
             WHERE C2.InvestmentID = D.InvestmentID
               AND C2.InvestorID = D.InvestorID
               AND (S2.Typename = 'Contribution: Investments' OR S2.Typename = 'Distribution: Return of Capital')
               AND C2.EffectiveDate <= D.EffectiveDate
             
             UNION ALL
             
             SELECT D2.Amount AS Amt
             FROM IA_Distribution D2
             LEFT JOIN IA_Subtype S2 ON D2.DistributionTypeID = S2.SubtypeUID
             WHERE D2.InvestmentID = D.InvestmentID
               AND D2.InvestorID = D.InvestorID
               AND S2.Typename = 'Distribution: Return of Capital'
               AND D2.EffectiveDate <= D.EffectiveDate
         ) AS Combined
        ) AS Cum_Amt,
        CASE
            WHEN S.Typename = 'Distribution: Return of Capital' THEN 'Y'
            ELSE 'N'
        END AS Capital,
        CASE
            WHEN S.Typename IN ('Distribution: Preferred Return', 'Distribution: Excess Cash Flow') THEN 'Y'
            ELSE 'N'
        END AS ROE_Income
    FROM 
        IA_Distribution D
    LEFT JOIN 
        IA_Subtype S 
        ON D.DistributionTypeID = S.SubtypeUID
        AND S.MajorType = 'Distribution'
)
SELECT *
FROM IA_Combined IC1
ORDER BY InvestmentID, InvestorID, EffectiveDate;